import asyncio
from datetime import datetime
import os
import uuid
import shutil
from typing import Union
from os import path, mkdir, getcwd, makedirs, listdir
from tqdm import tqdm
import keyboard
import zipfile
from collections import deque
import numpy as np
from threading import Thread
from queue import Queue
from windows import WindowCapture
from utils import FileWatcher, FixedRuntime, OsuSocketServer, ScreenRecorder, get_validated_input
from constants import CURRENT_STACK_NUM, FINAL_RESIZE_PERCENT, FRAME_DELAY, PLAY_AREA_CAPTURE_PARAMS
from tempfile import TemporaryDirectory


# list_window_names()


PROJECT_PATH: Union[str, None] = None


def start_capture():
    global FRAMES_TOTAL
    global FRAMES_PROCESSED
    global PROJECT_NAME
    global PROJECT_PATH
    with TemporaryDirectory() as CAPTURE_DIR:
        with OsuSocketServer(on_state_updated=lambda a: a) as osu_client:
            capture_frame = False

            PROJECT_NAME = get_validated_input(
                'What Would You Like To Name This Project ?:', conversion_fn=lambda a: a.lower().strip())

            PROJECT_PATH = path.join(
                getcwd(), 'data', 'raw', f"{PROJECT_NAME}.zip")

            def toggle_capture():
                nonlocal osu_client
                nonlocal capture_frame
                nonlocal CAPTURE_DIR
                global PROJECT_NAME

                if osu_client is None or not osu_client.active:
                    return

                capture_frame = not capture_frame
                if capture_frame:
                    osu_client.send(
                        f'save,{PROJECT_NAME},start,0.01,{CAPTURE_DIR}')
                    print(f'Capturing Frames')
                else:
                    osu_client.send(f'save,{PROJECT_NAME},stop,0.01, ')
                    print(f'Stopped Capturing Frames')

            keyboard.add_hotkey('\\', callback=toggle_capture)

            try:

                try:
                    print(
                        "Ready To Capture Frames, use the \"\\\" Key to toggle capture")
                    while True:
                        with FixedRuntime(target_time=10):
                            pass
                except KeyboardInterrupt as e:
                    osu_client.send(f'save,{PROJECT_NAME},stop,0.01, ')

                    with zipfile.ZipFile(PROJECT_PATH, "w", zipfile.ZIP_DEFLATED) as zip:
                        temp_files = listdir(CAPTURE_DIR)

                        for file in tqdm(temp_files, desc="Zipping up captured images"):

                            zip.write(os.path.join(CAPTURE_DIR, file), file)

            except Exception as e:
                osu_client.send(f'save,{PROJECT_NAME},stop,0.01, ')

                print(e)
