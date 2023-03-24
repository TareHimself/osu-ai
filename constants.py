import cv2
import numpy as np
import torch
from os import getcwd, path, makedirs
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

FINAL_RESIZE_PERCENT = 0.2

FINAL_PLAY_AREA_SIZE = (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
    PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT))

PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CURRENT_STACK_NUM = 4

FRAME_DELAY = 0.01

if not path.exists(path.join(getcwd(), 'data', 'raw')):
    makedirs(path.join(getcwd(), 'data', 'raw'))

if not path.exists(path.join(getcwd(), 'data', 'processed')):
    makedirs(path.join(getcwd(), 'data', 'processed'))
