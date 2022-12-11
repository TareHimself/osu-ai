import cv2
import numpy as np
import torch
from os import getcwd, path
from .windows import derive_capture_params

ASSETS_DIR = path.normpath(path.join(getcwd(), 'assets'))
MODELS_DIR = path.normpath(path.join(
    getcwd(), 'models'))
RAW_DATA_DIR = path.normpath(path.join(
    getcwd(), 'data', 'raw'))
PROCESSED_DATA_DIR = path.normpath(path.join(
    getcwd(), 'data', 'processed'))
BUTTON_CLICKED_COLOR = np.array([254, 254, 220])
BUTTON_CAPTURE_HEIGHT = 75
BUTTON_CAPTURE_WIDTH = 46
PLAY_AREA_CAPTURE_PARAMS = derive_capture_params()
PLAY_AREA_WIDTH_HEIGHT = np.array(
    [PLAY_AREA_CAPTURE_PARAMS[0], PLAY_AREA_CAPTURE_PARAMS[1]])
FINAL_RESIZE_PERCENT = 0.1
cursor_mat = cv2.imread(path.normpath(
    path.join(ASSETS_DIR, 'cursor.png')), cv2.IMREAD_COLOR)
GAME_CURSOR = cv2.resize(cursor_mat, (int(len(cursor_mat[0]) * FINAL_RESIZE_PERCENT), int(
    len(cursor_mat[1]) * FINAL_RESIZE_PERCENT)), interpolation=cv2.INTER_LINEAR)

PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
