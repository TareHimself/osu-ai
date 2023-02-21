import cv2
import numpy as np
import torch
from os import getcwd, path
from windows import derive_capture_params

ASSETS_DIR = path.normpath(path.join(getcwd(), 'assets'))
MODELS_DIR = path.normpath(path.join(
    getcwd(), 'models'))
RAW_DATA_DIR = path.normpath(path.join(
    getcwd(), 'data', 'raw'))
PROCESSED_DATA_DIR = path.normpath(path.join(
    getcwd(), 'data', 'processed'))

PLAY_AREA_CAPTURE_PARAMS = derive_capture_params()
PLAY_AREA_WIDTH_HEIGHT = np.array(
    [PLAY_AREA_CAPTURE_PARAMS[0], PLAY_AREA_CAPTURE_PARAMS[1]])

FINAL_RESIZE_PERCENT = 0.1

FINAL_PLAY_AREA_SIZE = (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
    PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT))

cursor_mat = cv2.imread(path.normpath(
    path.join(ASSETS_DIR, 'cursor.png')), cv2.IMREAD_COLOR)
# cv2.imshow("debug", mask_mat * cursor_mat)
# cv2.waitKey(0)

PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CURRENT_STACK_NUM = 3

FRAME_DELAY = 0.01
