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


class OsuAgentState:
    WAITING_FOR_MAP = "[WAITING FOR MAP]"
    WAITING_FOR_OPENING = "[WAITING FOR OP]"
    PLAYING_MAP = "[PLAYING]"


class OsuAgent:

    def __init__(self, stacks):
        self.stacks = stacks
        self.sock = OsuSocketServer(self.on_map_state_updated)
        self.hwnd = win32gui.FindWindow(None, "osu! (development)")
        self.buff = Queue()
        self.state = OsuAgentState.WAITING_FOR_MAP
        Thread(target=self.draw, group=None, daemon=True).start()

    def on_map_state_updated(self, state: str):

        if state == "MAP_BEGIN":
            self.update_state(OsuAgentState.WAITING_FOR_OPENING)

        if state == "MAP_END":
            self.update_state(OsuAgentState.WAITING_FOR_MAP)

    def update_state(self, newState: str):
        initial = self.state
        self.state = newState
        print("State changed from", initial, "To", self.state)

    def draw(self):
        # while True:
        #     stacked = self.buff.get()
        #     if stacked is None:
        #         break
        #     cv2.imshow("Debug", stacked.transpose(1, 2, 0))
        #     cv2.waitKey(100)
        pass

    def capture_frames(self, stack_num=3, stack_interval=0.01, resize: tuple[int, int] = (1920, 1080)):
        width = int(PLAY_AREA_CAPTURE_PARAMS[0])
        height = int(PLAY_AREA_CAPTURE_PARAMS[1])
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
        frames = []
        display = []
        for i in range(stack_num):
            cDC.SelectObject(dataBitMap)
            cDC.BitBlt((0, 0), (width, height), dcObj,
                       (PLAY_AREA_CAPTURE_PARAMS[2], PLAY_AREA_CAPTURE_PARAMS[3]), win32con.SRCCOPY)
            time.sleep(stack_interval)
            # convert the raw data into a format opencv can read
            #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
            signedIntsArray = dataBitMap.GetBitmapBits(True)
            frame = np.frombuffer(signedIntsArray, dtype='uint8')
            frame.shape = (height, width, 4)
            frame = np.ascontiguousarray(frame[..., :3])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display.append(frame)
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)

        frames = np.stack(frames)
        display = np.stack(display)
        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        return frames, display

    def get_state(self):
        if self.state == OsuAgentState.WAITING_FOR_MAP:
            while self.state == OsuAgentState.WAITING_FOR_MAP:
                time.sleep(0.01)

        while self.state == OsuAgentState.WAITING_FOR_OPENING:
            score, acc, game_time = asyncio.run(
                self.sock.send_and_wait("state", "0.0,0.0,-10.0")).split(',')
            if float(game_time) >= 0:
                self.update_state(OsuAgentState.PLAYING_MAP)
                break
            time.sleep(0.01)

        stacked, display_frame = self.capture_frames(stack_interval=0.01, stack_num=self.stacks, resize=(int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
            PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)))
        score, acc, game_time = asyncio.run(
            self.sock.send_and_wait("state", "0.0,0.0,0.0")).split(',')

        state_arr = [float(score), float(acc), float(game_time)]

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type()
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109

        self.buff.put(display_frame)
        return [stacked / 255] + state_arr

    def do_action(self, action):
        if action < 0.5:
            keyboard.release("z")
        else:
            keyboard.press("z")

    def reset(self):
        keyboard.press_and_release("`")
        time.sleep(0.01)
        print("State at reset", self.state)
        while self.state == OsuAgentState.WAITING_FOR_MAP:
            time.sleep(0.01)

    def kill(self):
        self.buff.put(None)
        self.sock.kill()
