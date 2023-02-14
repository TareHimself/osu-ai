from threading import Thread
import time
import cv2
import win32gui
import win32ui
import win32con
import numpy as np
from queue import Queue


class WindowCapture:
    def __init__(self, window_name=None) -> None:
        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            self.hwnd = win32gui.FindWindow(None, window_name)
            if not self.hwnd:
                WindowCapture.list_window_names()
                raise Exception(
                    f"Window '{window_name}' Not Found Select window name from above")

    def capture(self, width=1920, height=1080, dx=0, dy=0) -> np.ndarray:
        width = int(width)
        height = int(height)
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (width, height), dcObj, (dx, dy), win32con.SRCCOPY)

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
        img = np.ascontiguousarray(img)

        return img

    def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)


class WindowStream:
    def __init__(self, window_name=None, width=1920, height=1080, dx=0, dy=0) -> None:
        self.window_name = window_name
        self.width = width
        self.height = height
        self.dx = dx
        self.dy = dy
        self.frame_buffer = Queue()
        self.frames = 0
        Thread(daemon=True, group=None, target=self.capture).start()
        Thread(daemon=True, group=None, target=self.do_frame_rate).start()
        Thread(daemon=True, group=None, target=self.stream).start()

    def do_frame_rate(self):
        while True:
            time.sleep(1)
            print(f"Capture FPS {self.frames:.0f}       ", end="\r")
            self.frames = 0

    def capture(self):
        window_capture = WindowCapture(self.window_name)
        while True:
            frame = window_capture.capture(
                self.width, self.height, self.dx, self.dy)
            self.frame_buffer.put(frame)
            self.frames += 1

    def stream(self):
        while True:
            frame = self.frame_buffer.get()
            if frame is not None:
                cv2.imshow(
                    f"Stream of window {self.window_name if self.window_name is not None else 'Desktop'}", frame)
                cv2.waitKey(1)


def derive_capture_params(window_width=1920, window_height=1080, capture_height=1000):
    osu_playfield_ratio = 3/4

    capture_width = int(capture_height / osu_playfield_ratio)
    capture_params = [capture_width, capture_height,
                      int((window_width-capture_width)/2), int((window_height-(capture_height))/2)]

    return capture_params


def coordinates_to_playfield_percent(window_width=1920, window_height=1080, capture_height=1000):
    osu_playfield_ratio = 3/4

    capture_width = int(capture_height / osu_playfield_ratio)
    capture_params = [capture_width, capture_height,
                      int((window_width-capture_width)/2), int((window_height-(capture_height))/2)]

    return capture_params
