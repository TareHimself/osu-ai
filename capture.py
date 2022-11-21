from typing import Union
from os import path, mkdir, getcwd
import time
import keyboard
import cv2
import numpy as np
from threading import Thread
from queue import Queue
from windows import derive_capture_params, WindowCapture, WindowStream


# list_window_names()
ScreenShot = WindowCapture("osu!")

capture_frame = False


def toggle_capture():
    global capture_frame
    capture_frame = not capture_frame


keyboard.add_hotkey('shift+r', callback=toggle_capture)

buffer = Queue()

FINAL_RESIZE_PERCENT = 0.3

CAPTURE_PARAMS = derive_capture_params()
RESIZED_PARAMS = derive_capture_params()
KEYS_CAPTURE_PARAMS = [46 + 46, 75, 1761, 960]

# WindowStream('osu!', *CAPTURE_PARAMS)
print(CAPTURE_PARAMS)

project_name = input(
    'What Would You Like To Name This Project ?:').lower().strip()

PROJECT_PATH = path.join(getcwd(), 'data', 'raw', project_name)

try:
    mkdir(PROJECT_PATH)
except:
    pass

count = 0


def process_frames():
    global count
    global last_left
    global last_right
    global clicks
    stop_saving = False

    while not stop_saving:
        frame: Union[np.ndarray, None] = buffer.get(block=True)
        if frame is None:
            stop_saving = True
            continue

        cv2.imwrite(path.join(
            PROJECT_PATH, f'{count}.png'), frame)
        count += 1
        print(
            f'Processed {count} frames :: {buffer.qsize()} Remaining', end='\r')


save_thread = Thread(group=None, target=process_frames)
save_thread.start()

try:
    while True:
        start = time.time()
        if capture_frame:
            frame = ScreenShot.capture()
            if frame is not None:
                buffer.put(frame)

        elapsed = time.time() - start
        wait_time = 0.01 - elapsed
        if wait_time > 0:
            time.sleep(wait_time)


except KeyboardInterrupt as e:

    buffer.put(None)
    save_thread.join()
    pass
