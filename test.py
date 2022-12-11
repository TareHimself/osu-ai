
import cv2
import numpy as np
from datetime import datetime

file_name = datetime.utcnow().strftime("%y%m%d%H%M%S%f")
TEST_DATA_PATH = "D:\Github\osu-ai\data\\raw\circles\\3932.png"

image = cv2.imread(TEST_DATA_PATH)
print()
# cv2.imshow("debug", get_buttons_from_screenshot(image))
# cv2.waitKey(0)
