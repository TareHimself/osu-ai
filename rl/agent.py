import numpy as np
import win32gui
import win32ui
import win32con
import asyncio
import cv2
import keyboard
import time
import pygame
from queue import Queue
from torchvision import transforms
from utils import OsuSocketServer
from collections import deque
from threading import Thread
from constants import PLAY_AREA_CAPTURE_PARAMS, FINAL_RESIZE_PERCENT, PYTORCH_DEVICE

ConvertToTensor = transforms.ToTensor()

MAX_MEMORY = 100_000
BATCH_SIZE = 32
LR = 0.001


class OsuAgent:

    def __init__(self):
        self.sock = OsuSocketServer()
        self.hwnd = win32gui.FindWindow(None, "osu! (development)")
        self.buff = Queue()
        Thread(target=self.draw, group=None, daemon=True).start()

    def draw(self):
        while True:
            frame = self.buff.get()
            if frame is None:
                break

            cv2.imshow("Debug", frame)
            cv2.waitKey(1)

    def get_state(self):

        width = int(PLAY_AREA_CAPTURE_PARAMS[0])
        height = int(PLAY_AREA_CAPTURE_PARAMS[1])
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (width, height), dcObj,
                   (PLAY_AREA_CAPTURE_PARAMS[2], PLAY_AREA_CAPTURE_PARAMS[3]), win32con.SRCCOPY)

        state_arr = [0, 0, -20]

        while state_arr[2] <= 0.0:
            response = asyncio.run(self.sock.send_and_wait(
                "state")) if self.sock.client is not None else "0,0.0,0.0"

            state_from_game: str = response if response != "NO_MAP" else "0,0.0,0.0"

            state_from_game = list(map(float, state_from_game.split(",")))

            state_arr = state_from_game

            time.sleep(0.01)

        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type()
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[..., :3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109

        img = cv2.resize(np.ascontiguousarray(img), (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
            PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)), interpolation=cv2.INTER_LINEAR)

        self.buff.put(img)

        img = ConvertToTensor(img / 255).numpy().astype(np.float32)

        return [img] + state_arr

    def do_action(self, action):
        if action < 0.5:
            keyboard.release("z")
        else:
            keyboard.press("z")

    def reset(self):
        keyboard.press_and_release("`")
        time.sleep(0.01)
        state_arr = [0, 0, -20]

        while state_arr[2] <= 0.0:
            response = asyncio.run(self.sock.send_and_wait(
                "state")) if self.sock.client is not None else "0,0.0,0.0"

            state_from_game: str = response if response != "NO_MAP" else "0,0.0,0.0"

            state_from_game = list(map(float, state_from_game.split(",")))

            state_arr = state_from_game
            time.sleep(0.01)

    def kill(self):
        self.buff.put(None)
        self.sock.kill()
