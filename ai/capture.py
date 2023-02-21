import asyncio
from datetime import datetime
import os
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
from utils import FileWatcher, FixedRuntime, OsuSocketServer, get_validated_input
from constants import CURRENT_STACK_NUM, FINAL_RESIZE_PERCENT, FRAME_DELAY, PLAY_AREA_CAPTURE_PARAMS

# list_window_names()


PROJECT_PATH: Union[str, None] = None


def start_capture():
    global FRAMES_TOTAL
    global FRAMES_PROCESSED
    global PROJECT_NAME
    global PROJECT_PATH

    FRAMES_PROCESSED = 0
    FRAMES_TOTAL = 0

    capture_frame = False

    PROJECT_NAME = get_validated_input(
        'What Would You Like To Name This Project ?:', conversion_fn=lambda a: a.lower().strip())

    PROJECT_PATH = path.join(getcwd(), 'data', 'raw', PROJECT_NAME)

    if path.exists(PROJECT_PATH):
        shutil.rmtree(PROJECT_PATH)
    makedirs(PROJECT_PATH)

    socket_server = OsuSocketServer(on_state_updated=lambda a: a)

    def toggle_capture():
        nonlocal socket_server
        nonlocal capture_frame
        global PROJECT_NAME
        capture_frame = not capture_frame
        if capture_frame:
            socket_server.send(f'save,{PROJECT_NAME},start,0.01')
            print(f'Capturing Frames')
        else:
            socket_server.send(f'save,{PROJECT_NAME},stop,0.01')
            print(f'Stopped Capturing Frames')

    keyboard.add_hotkey('\\', callback=toggle_capture)

    try:

        try:
            print("Ready To Capture Frames")
            while True:
                with FixedRuntime(target_time=10):
                    pass
        except KeyboardInterrupt as e:
            socket_server.send(f'save,{PROJECT_NAME},stop,0.01')
            try:
                PENDING_CAPTURE_PATH = path.join(getcwd(), "pending-capture")
                for file in os.listdir(PENDING_CAPTURE_PATH):
                    if file.split('-')[0].strip() == PROJECT_NAME:
                        os.rename(path.join(PENDING_CAPTURE_PATH, file),
                                  path.join(PROJECT_PATH, file))
            except Exception as e:
                print("Error moving capture data", e)

            socket_server.kill()
    except Exception as e:
        socket_server.send(f'save,{PROJECT_NAME},stop,0.01')
        socket_server.kill()
        print(e)
