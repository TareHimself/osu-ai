import time
import cv2
from collections import deque
from windows import WindowCapture

cap = WindowCapture("osu!")
STACK_NUM = 3
frame_history = deque(maxlen=3)
while True:
    frame, stacked = cap.capture(list(frame_history), STACK_NUM)
    frame_history.append(frame)
    cv2.imshow("Debug", stacked)
    cv2.waitKey(1)
