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
        self.num_captured = 0
        self.wDC = None
        self.dcObj = None
        self.cDC = None
        self.dataBitMap = None

    def ensure_resources(self, width: int, height: int):
        if self.num_captured > 50:
            self.dcObj.DeleteDC()
            self.cDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, self.wDC)
            win32gui.DeleteObject(self.dataBitMap.GetHandle())
            self.wDC = None
            self.dcObj = None
            self.cDC = None
            self.dataBitMap = None
            self.num_captured = 0

        if self.wDC is None:
            self.wDC = win32gui.GetWindowDC(self.hwnd)
            self.dcObj = win32ui.CreateDCFromHandle(self.wDC)
            self.cDC = self.dcObj.CreateCompatibleDC()
            self.dataBitMap = win32ui.CreateBitmap()
            self.dataBitMap.CreateCompatibleBitmap(self.dcObj, width, height)
        self.num_captured += 1

    def get_frame(self, resize: tuple[int, int], width: int, height: int, dx: int, dy: int, is_stacking=False):
        self.ensure_resources(width, height)

        self.cDC.SelectObject(self.dataBitMap)
        self.cDC.BitBlt((0, 0), (width, height), self.dcObj,
                        (dx, dy), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = self.dataBitMap.GetBitmapBits(True)
        frame = np.frombuffer(signedIntsArray, dtype='uint8')
        frame.shape = (height, width, 4)
        frame = np.ascontiguousarray(frame[..., :3])
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
        if not is_stacking:
            return frame

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def capture(self, prev_frames=[], stack_num=0, resize: tuple[int, int] = (1920, 1080), width=1920, height=1080, dx=0, dy=0) -> np.ndarray:
        width = int(width)
        height = int(height)

        self.ensure_resources(width, height)

        is_stacking = stack_num != 0
        frame = self.get_frame(resize, width, height, dx, dy, is_stacking)

        if not is_stacking:
            return frame

        prev_count = len(prev_frames)
        needed_count = stack_num - prev_count
        final_frames = []
        if needed_count > 1:
            final_frames = prev_frames + [frame for _ in range(needed_count)]
        else:
            final_frames = prev_frames[prev_count -
                                       (stack_num - 1):prev_count] + [frame]

        return [frame, np.stack(final_frames)]

    # Deleting (Calling destructor)
    def __del__(self):
        if self.wDC is not None:
            self.dcObj.DeleteDC()
            self.cDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, self.wDC)
            win32gui.DeleteObject(self.dataBitMap.GetHandle())
            self.wDC = None
            self.dcObj = None
            self.cDC = None
            self.dataBitMap = None
            self.num_captured = 0

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
