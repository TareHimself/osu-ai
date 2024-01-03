import torch
import screeninfo
from os import getcwd, path, makedirs

SCREEN_WIDTH = screeninfo.get_monitors()[0].width
SCREEN_HEIGHT = screeninfo.get_monitors()[0].height
ASSETS_DIR = path.normpath(path.join(getcwd(), "assets"))
MODELS_DIR = path.normpath(path.join(getcwd(), "models"))
RAW_DATA_DIR = path.normpath(path.join(getcwd(), "data", "raw"))
PROCESSED_DATA_DIR = path.normpath(path.join(getcwd(), "data", "processed"))

CAPTURE_HEIGHT_PERCENT = 1.0


PLAY_AREA_RATIO = 4 / 3

FINAL_IMAGE_WIDTH = 96  # int(1920 * 0.1)
# FINAL_IMAGE_WIDTH = 80
FINAL_IMAGE_HEIGHT = int(FINAL_IMAGE_WIDTH / PLAY_AREA_RATIO)

FINAL_PLAY_AREA_SIZE = (FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT)

PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CURRENT_STACK_NUM = 4

FRAME_DELAY = 0.01

MAX_THREADS_FOR_RESIZING = 20

if not path.exists(RAW_DATA_DIR):
    makedirs(RAW_DATA_DIR)

if not path.exists(PROCESSED_DATA_DIR):
    makedirs(PROCESSED_DATA_DIR)
