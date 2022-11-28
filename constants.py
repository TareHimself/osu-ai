import cv2
import numpy as np
from windows import derive_capture_params

BUTTON_CLICKED_COLOR = np.array([254, 254, 220])
BUTTON_CAPTURE_HEIGHT = 75
BUTTON_CAPTURE_WIDTH = 46
PLAY_AREA_CAPTURE_PARAMS = derive_capture_params()
PLAY_AREA_WIDTH_HEIGHT = np.array([PLAY_AREA_CAPTURE_PARAMS[0],PLAY_AREA_CAPTURE_PARAMS[1]])
FINAL_RESIZE_PERCENT = 0.3
GAME_CURSOR = cv2.imread('cursor.png', cv2.IMREAD_COLOR)
