import keyboard
from windows import WindowCapture
import time
import cv2
import numpy as np
import torch
from utils import get_models, get_validated_input, load_model_data
import torchvision.transforms as transforms
from ai.models import ActionsNet, AimNet
from constants import FINAL_RESIZE_PERCENT, PLAY_AREA_CAPTURE_PARAMS, PYTORCH_DEVICE
import win32api

transform = transforms.ToTensor()

do_prediction = False

KEYS_STATE_TO_STRING = {
    0: "Idle    ",
    1: "Button 1",
    2: "Button 2"
}

last_state = 0


def set_key_state(state: int):
    global last_state
    if state == 0:
        keyboard.release('x')
        keyboard.release('z')
    elif state == 1:
        keyboard.release('z')
        keyboard.press('x')
    elif state == 2:
        keyboard.release('x')
        keyboard.press('z')

    last_state = state


def toggle_capture():
    global do_prediction
    do_prediction = not do_prediction


keyboard.add_hotkey('\\', callback=toggle_capture)


def start_play(time_between_frames=0):
    global do_prediction
    try:
        do_prediction = False

        window_capture = WindowCapture("osu! (development)")

        action_models = get_models('model_action_')

        aim_models = get_models('model_aim_')

        user_choice = get_validated_input(f"""What type of model would you like to test?
    [0] Actions Model | {len(action_models)} Available
    [1] Aim Model | {len(aim_models)} Available
    [2] Both Models
""", lambda a: a.strip().isnumeric() and (0 <= int(a.strip()) <= 2), lambda a: int(a.strip()))

        action_model_index = None
        aim_model_index = None

        if user_choice == 0 or user_choice == 2:
            prompt = "What actions model would you like to use?\n"
            for i in range(len(action_models)):
                prompt += f"    [{i}] {action_models[i]}\n"

            action_model_index = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                0 <= int(a.strip()) < len(action_models)), lambda a: int(a.strip()))

        if user_choice == 1 or user_choice == 2:
            prompt = "What aim model would you like to use?\n"
            for i in range(len(aim_models)):
                prompt += f"    [{i}] {aim_models[i]}\n"

            aim_model_index = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                0 <= int(a.strip()) < len(aim_models)), lambda a: int(a.strip()))

        actions_model = None
        aim_model = None

        if action_model_index is not None:
            actions_model_data = load_model_data(
                action_models[action_model_index])
            actions_model = ActionsNet().to(PYTORCH_DEVICE)
            actions_model.load_state_dict(actions_model_data['state'])
            actions_model.eval()

        if aim_model_index is not None:
            aim_model_data = load_model_data(aim_models[aim_model_index])
            aim_model = AimNet().to(PYTORCH_DEVICE)
            aim_model.load_state_dict(aim_model_data['state'])
            aim_model.eval()

        print("Configuration Ready,Press 'Shift + R' To Toggle the model(s)")
        fps = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        try:
            while True:
                start = time.time()
                debug = f"FPS {sum(fps) / len(fps):.4f} "
                frame = window_capture.capture(*PLAY_AREA_CAPTURE_PARAMS)
                if frame is not None:
                    cv_img = cv2.resize(
                        frame, (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
                            PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)), interpolation=cv2.INTER_LINEAR)

                    # cv2.imshow("Debug", cv_img)
                    # cv2.waitKey(1) np.stack([cv_img, cv_img, cv_img], axis=-1)
                    stacked = cv_img / 255

                    if do_prediction:
                        converted_frame = transform(stacked)

                        inputs = converted_frame.reshape(
                            (1, converted_frame.shape[0], converted_frame.shape[1], converted_frame.shape[2])).type(
                            torch.FloatTensor).to(PYTORCH_DEVICE)

                        if aim_model:
                            output = aim_model(inputs)
                            mouse_x_percent, mouse_y_percent = output[0]
                            position = (int((mouse_x_percent * PLAY_AREA_CAPTURE_PARAMS[0]) + PLAY_AREA_CAPTURE_PARAMS[2]), int(
                                (mouse_y_percent * PLAY_AREA_CAPTURE_PARAMS[1]) + PLAY_AREA_CAPTURE_PARAMS[3]))
                            win32api.SetCursorPos(position)
                            debug += f"Cursor Position {position}        "

                        if actions_model:
                            output = actions_model(inputs)
                            _, predicated = torch.max(output, dim=1)

                            probs = torch.softmax(output, dim=1)
                            prob = probs[0][predicated.item()]
                            debug += f"Decision {KEYS_STATE_TO_STRING[predicated.item()]} :: chance {prob.item():.2f}       "

                            if prob.item() > 0:  # 0.7:
                                set_key_state(predicated.item())

                fps.append(1 / (time.time() - start))
                fps.pop(0)
                print(debug, end='\r')
        except KeyboardInterrupt as e:
            pass
    except Exception as e:
        print(e)
