from windows import WindowCapture
import cv2
from ai.utils import FixedRuntime
from collections import deque
cap = WindowCapture()
previous = deque(maxlen=3)
result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         20, (1920, 1080))
try:
    while True:
        with FixedRuntime(0.0167):
            frame, stacked = cap.capture(
                prev_frames=list(previous), stack_num=3)
            stacked = stacked.transpose(1, 2, 0)
            result.write(stacked)
            previous.append(frame)
            cv2.imshow("Debug", stacked)
            cv2.waitKey(1)

except KeyboardInterrupt:
    result.release()
