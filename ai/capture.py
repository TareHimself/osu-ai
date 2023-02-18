import asyncio
from datetime import datetime
import shutil
from typing import Union
from os import path, mkdir, getcwd, makedirs
import time
import keyboard
import cv2
from collections import deque
import numpy as np
from threading import Thread
from queue import Queue
from windows import WindowCapture
from utils import FileWatcher, OsuSocketServer, get_validated_input
from constants import FINAL_RESIZE_PERCENT, PLAY_AREA_CAPTURE_PARAMS

# list_window_names()

FRAME_BUFFER: Union[Queue, None] = None

PROJECT_NAME: Union[str, None] = None

IMAGES_PATH: Union[str, None] = None

STATE_PATH: Union[str, None] = None

FRAMES_PROCESSED = 0

FRAMES_TOTAL = 0


def process_frames_in_background():
    global FRAMES_PROCESSED
    global FRAMES_TOTAL
    global FRAME_BUFFER
    global IMAGES_PATH
    global STATE_PATH

    while True:
        frame: Union[Union[np.ndarray, str],
                     None] = FRAME_BUFFER.get(block=True)
        if frame is None:
            break

        sct, state = frame

        filename = f"{datetime.utcnow().strftime('%y%m%d%H%M%S%f')}"

        np.save(path.join(IMAGES_PATH,
                f"{filename}"), sct, allow_pickle=True)

        with open(path.join(STATE_PATH, f"{filename}.txt"), "w") as data:
            data.write(state)

        FRAMES_PROCESSED += 1
        print(
            f'Processed {FRAMES_PROCESSED} frames :: {FRAME_BUFFER.qsize()} Remaining          ', end='\r')


def start_capture():
    global FRAMES_TOTAL
    global FRAMES_PROCESSED
    global PROJECT_NAME
    global IMAGES_PATH
    global FRAME_BUFFER
    global STATE_PATH

    FRAMES_PROCESSED = 0
    FRAMES_TOTAL = 0

    capture_frame = False

    def toggle_capture():
        nonlocal capture_frame
        capture_frame = not capture_frame

    keyboard.add_hotkey('\\', callback=toggle_capture)

    FRAME_BUFFER = Queue()

    PROJECT_NAME = get_validated_input(
        'What Would You Like To Name This Project ?:', conversion_fn=lambda a: a.lower().strip())

    IMAGES_PATH = path.join(getcwd(), 'data', 'raw', PROJECT_NAME, 'display')

    STATE_PATH = path.join(getcwd(), 'data', 'raw', PROJECT_NAME, 'state')

    FRAME_BUFF_MAX = 3

    FRAME_BUFF = deque(maxlen=FRAME_BUFF_MAX)

    if path.exists(IMAGES_PATH):
        shutil.rmtree(IMAGES_PATH)
    makedirs(IMAGES_PATH)

    if path.exists(STATE_PATH):
        shutil.rmtree(STATE_PATH)
    makedirs(STATE_PATH)

    save_thread = Thread(group=None, target=process_frames_in_background)

    socket_server = OsuSocketServer(on_state_updated=lambda a: a)

    try:
        start = time.time()
        print(
            f'Processed {FRAMES_PROCESSED} frames :: {FRAME_BUFFER.qsize()} Remaining          ', end='\r')
        MIN_FRAME_DELAY = 0.01
        save_thread.start()
        window_capture = WindowCapture("osu! (development)")
        try:
            while True:
                start = time.time()
                if capture_frame:
                    frame, stacked = window_capture.capture(
                        list(FRAME_BUFF), FRAME_BUFF_MAX, (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
                            PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)), *PLAY_AREA_CAPTURE_PARAMS)
                    FRAME_BUFF.append(frame)

                    data = asyncio.run(
                        socket_server.send_and_wait('cap'))
                    FRAME_BUFFER.put(
                        [stacked, data])
                    FRAMES_TOTAL += 1

                elapsed = time.time() - start
                wait_time = MIN_FRAME_DELAY - elapsed
                if wait_time > 0:
                    time.sleep(wait_time)

        except KeyboardInterrupt as e:

            FRAME_BUFFER.put(None)
            save_thread.join()
            save_thread = None

            socket_server.kill()
    except Exception as e:
        if save_thread is not None and save_thread.is_alive():
            FRAME_BUFFER.put(None)
            save_thread.join()

        socket_server.kill()
        print(e)
