from datetime import datetime
import shutil
from typing import Union
from os import path, mkdir, getcwd, makedirs
import time
import keyboard
import cv2
import numpy as np
from threading import Thread
from queue import Queue
from windows import WindowCapture
from utils import get_validated_input
from ai.dataset import get_resized_play_area, get_buttons_from_screenshot

# list_window_names()

FRAME_BUFFER: Union[Queue, None] = None

PROJECT_NAME: Union[str, None] = None

IMAGES_PATH: Union[str, None] = None

BUTTONS_PATH: Union[str, None] = None

FRAMES_PROCESSED = 0

FRAMES_TOTAL = 0


def process_frames_in_background():
    global FRAMES_PROCESSED
    global FRAMES_TOTAL
    global FRAME_BUFFER
    global IMAGES_PATH
    global BUTTONS_PATH

    while True:
        frame: Union[np.ndarray, None] = FRAME_BUFFER.get(block=True)
        if frame is None:
            break

        filename = f"{datetime.utcnow().strftime('%y%m%d%H%M%S%f')}.png"

        cv2.imwrite(
            path.join(IMAGES_PATH, filename), get_resized_play_area(frame))
        cv2.imwrite(path.join(
            BUTTONS_PATH, filename), get_buttons_from_screenshot(frame))
        FRAMES_PROCESSED += 1
        print(
            f'Processed {FRAMES_PROCESSED} frames :: {FRAME_BUFFER.qsize()} Remaining          ', end='\r')


def start_capture():
    global FRAMES_TOTAL
    global FRAMES_PROCESSED
    global PROJECT_NAME
    global IMAGES_PATH
    global FRAME_BUFFER
    global BUTTONS_PATH

    FRAMES_PROCESSED = 0
    FRAMES_TOTAL = 0

    capture_frame = False

    def toggle_capture():
        nonlocal capture_frame
        capture_frame = not capture_frame

    keyboard.add_hotkey('shift+r', callback=toggle_capture)

    FRAME_BUFFER = Queue()

    PROJECT_NAME = get_validated_input(
        'What Would You Like To Name This Project ?:', conversion_fn=lambda a: a.lower().strip())

    IMAGES_PATH = path.join(getcwd(), 'data', 'raw', PROJECT_NAME, 'area')

    BUTTONS_PATH = path.join(getcwd(), 'data', 'raw', PROJECT_NAME, 'buttons')

    if path.exists(IMAGES_PATH):
        shutil.rmtree(IMAGES_PATH)
    makedirs(IMAGES_PATH)

    if path.exists(BUTTONS_PATH):
        shutil.rmtree(BUTTONS_PATH)
    makedirs(BUTTONS_PATH)

    save_thread = Thread(group=None, target=process_frames_in_background)

    try:
        print(
            f'Processed {FRAMES_PROCESSED} frames :: {FRAME_BUFFER.qsize()} Remaining          ', end='\r')
        save_thread.start()
        window_capture = WindowCapture("osu!")
        try:
            while True:
                start = time.time()
                if capture_frame:
                    frame = window_capture.capture()
                    if frame is not None:
                        FRAME_BUFFER.put(frame)
                        FRAMES_TOTAL += 1

                # elapsed = time.time() - start
                # wait_time = 0.01 - elapsed
                # if wait_time > 0:
                #     time.sleep(wait_time)

        except KeyboardInterrupt as e:

            FRAME_BUFFER.put(None)
            save_thread.join()
            save_thread = None
    except Exception as e:
        if save_thread is not None and save_thread.is_alive():
            FRAME_BUFFER.put(None)
            save_thread.join()
        print(e)
