import os
from threading import Thread
import time
import traceback
import cv2
import keyboard
import win32api

from constants import FINAL_RESIZE_PERCENT, PLAY_AREA_CAPTURE_PARAMS
from windows import WindowCapture


class WatchFileContent(Thread):
    def __init__(self, file_path, callback):
        super().__init__(group=None, daemon=True)
        self.file_path = file_path
        self.callback = callback
        self.callback(open(self.file_path).readlines())

    def run(self):
        modifiedOn = os.path.getmtime(self.file_path)
        try:
            while True:
                time.sleep(0.05)
                modified = os.path.getmtime(self.file_path)
                if modified != modifiedOn:
                    modifiedOn = modified
                    self.callback(open(self.file_path).readlines())
        except Exception as e:
            print(traceback.format_exc())


class OsuPlayer:
    def __init__(self):
        self.accuracy = 1
        self.cap = WindowCapture("osu!")
        self.watcher = WatchFileContent(
            r'C:\Users\Taree\Pictures\accuracy.txt', self.on_accuracy_updated)

    def on_accuracy_updated(self, a: list[str]):
        self.accuracy = int(a[0])

    def do_action(self, mouse_x_percent, mouse_y_percent, key_state):

        mouse_x_scaled = PLAY_AREA_CAPTURE_PARAMS[2] + \
            (mouse_x_percent * PLAY_AREA_CAPTURE_PARAMS[0])
        mouse_y_scaled = PLAY_AREA_CAPTURE_PARAMS[3] + \
            (mouse_y_percent * PLAY_AREA_CAPTURE_PARAMS[1])

        win32api.SetCursorPos((mouse_x_scaled,  mouse_y_scaled))
        if key_state >= 0.5:
            keyboard.press('z')
        else:
            keyboard.release('z')

    def get_game(self):
        return cv2.resize(self.cap.capture(*PLAY_AREA_CAPTURE_PARAMS), (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
            PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)), interpolation=cv2.INTER_LINEAR)

    def get_accuracy(self):
        return self.accuracy

    def reset_game(self):
        self.accuracy = 1
        keyboard.press('`')
        time.sleep(0.5)
        keyboard.release('`')
        time.sleep(0.5)
        keyboard.press_and_release('spacebar')
