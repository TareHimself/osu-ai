from ai.constants import PLAY_AREA_CAPTURE_PARAMS
import numpy as np
import win32gui
import win32ui
import win32con
import time
import cv2

hwnd = win32gui.GetDesktopWindow()

stack_num = 10
stack_interval = 0.001

width = int(PLAY_AREA_CAPTURE_PARAMS[0])
height = int(PLAY_AREA_CAPTURE_PARAMS[1])
wDC = win32gui.GetWindowDC(hwnd)
dcObj = win32ui.CreateDCFromHandle(wDC)
cDC = dcObj.CreateCompatibleDC()
dataBitMap = win32ui.CreateBitmap()
dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
frames = []

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
    print(frame.shape, frame.dtype)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(frame.shape)

    frames.append(frame)

# free resources
dcObj.DeleteDC()
cDC.DeleteDC()
win32gui.ReleaseDC(hwnd, wDC)
win32gui.DeleteObject(dataBitMap.GetHandle())
print("Before stack", frames[0].shape)

final = np.stack(frames)
print("After stack", final[0].shape)

cv2.imshow('Debug', final[len(final) - 3: len(final)].transpose(1, 2, 0))
cv2.waitKey(10000)
# cv2.imwrite('debug.png', )
print(final.shape, final[0].shape, width, height)

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])
c = np.stack([a, a, b, b])
print(c.shape, c[0].shape, a.shape, b.shape)
