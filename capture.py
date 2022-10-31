import pyautogui
from os import path, mkdir, getcwd
import time
import keyboard
import cv2
import mss
import numpy
from threading import Thread
from queue import Queue

key_1_state = 0
key_2_state = 0

capture_frame = False


def on_press_1(e):
    global key_1_state
    key_1_state = 1


def on_release_1(e):
    global key_1_state
    key_1_state = 0


def on_press_2(e):
    global key_2_state
    key_2_state = 1


def on_release_2(e):
    global key_2_state
    key_2_state = 0


def toggle_capture():
    global capture_frame
    capture_frame = not capture_frame


keyboard.on_press_key(key='z', callback=on_press_1)
keyboard.on_press_key(key='x', callback=on_press_2)
keyboard.on_release_key(key='z', callback=on_release_1)
keyboard.on_release_key(key='x', callback=on_release_2)

keyboard.add_hotkey('ctrl+shift+t', callback=toggle_capture)

buffer = Queue()
SCALE_FACTOR = 0.25
FINAL_SIZE = (int(1920 * SCALE_FACTOR), int(1080 * SCALE_FACTOR))

project_name = input(
    'What Would You Like To Name This Project ?:').lower().strip()

PROJECT_PATH = path.join(getcwd(), 'data', 'raw', project_name)

mkdir(PROJECT_PATH)

count = 0


def SaveFrames():
    global count
    stop_saving = False

    while not stop_saving:
        data = buffer.get(block=True)
        if data is None:
            stop_saving = True
            continue

        frame, x, y, k1, k2 = data
        img = cv2.resize(numpy.array(
            frame), FINAL_SIZE, interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(path.join(
            PROJECT_PATH, '{}_{}_{}_{}_{}_{}.png'.format(project_name, count, x, y, k1, k2)), img)
        count += 1


saver = Thread(group=None, target=SaveFrames)
saver.start()

try:
    with mss.mss() as sct:
        while True:
            if capture_frame:
                frame = sct.grab(sct.monitors[1])
                if frame is not None:
                    mouseX, mouseY = pyautogui.position()
                    if mouseX >= 0 and mouseY >= 0:
                        buffer.put([frame, int(
                            mouseX * SCALE_FACTOR), int(mouseY * SCALE_FACTOR), key_1_state, key_2_state])
                    print('Saved {} frames'.format(count), end='\r')
            time.sleep(0.005)

except KeyboardInterrupt as e:
    buffer.put(None)
    saver.join()
    pass


# image.fromarray().convert('L').show()
# pyscreeze.screenshot('test.png')
# pyautogui.displayMousePosition()
