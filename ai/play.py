import keyboard
from windows import WindowCapture
import time
import cv2
import numpy as np
import torch
from collections import deque
from utils import FixedRuntime, get_models, get_validated_input
import torchvision.transforms as transforms
from ai.models import ActionsNet, AimNet
from constants import FINAL_RESIZE_PERCENT, PLAY_AREA_CAPTURE_PARAMS, PYTORCH_DEVICE
from ai.eval import ActionsThread, AimThread
import traceback

def start_play(time_between_frames=0):
    global do_prediction
    try:
        do_prediction = False

        action_models = get_models('action')

        aim_models = get_models('aim')

        user_choice = get_validated_input(f"""What type of model would you like to test?
    [0] Aim Model | {len(aim_models)} Available
    [1] Actions Model | {len(action_models)} Available
    [2] Both Models
""", lambda a: a.strip().isnumeric() and (0 <= int(a.strip()) <= 2), lambda a: int(a.strip()))

        action_model_index = None
        aim_model_index = None

        if user_choice == 0 or user_choice == 2:
            prompt = "What aim model would you like to use?\n"
            for i in range(len(aim_models)):
                prompt += f"    [{i}] {aim_models[i]}\n"

            aim_model_index = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                0 <= int(a.strip()) < len(aim_models)), lambda a: int(a.strip()))

        if user_choice == 1 or user_choice == 2:
            prompt = "What actions model would you like to use?\n"
            for i in range(len(action_models)):
                prompt += f"    [{i}] {action_models[i]}\n"

            action_model_index = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                0 <= int(a.strip()) < len(action_models)), lambda a: int(a.strip()))

        aim_model = None

        actions_model = None

        if aim_model_index is not None:
            aim_model = AimThread(model_id=aim_models[aim_model_index][0])

        if action_model_index is not None:
            actions_model = ActionsThread(
                model_id=action_models[action_model_index][0])

        try:
            while True:
                with FixedRuntime(2):
                    pass

        except KeyboardInterrupt as e:
            if actions_model is not None:
                actions_model.kill()
            if aim_model is not None:
                aim_model.kill()
    except Exception as e:
        print(traceback.format_exc())
