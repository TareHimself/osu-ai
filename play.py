from os import listdir, path, mkdir, getcwd
from queue import Queue
import keyboard
from windows import WindowCapture, derive_capture_params, WindowStream
import time
from os import path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from threading import Thread
import torchvision.transforms as transforms
from models import ClicksNet, MouseNet
from constants import FINAL_RESIZE_PERCENT, PLAY_AREA_CAPTURE_PARAMS
import win32api
transform = transforms.ToTensor()

cap = WindowCapture("osu!")


do_prediction = False


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


models = listdir(path.normpath(path.join(
    getcwd(), 'models')))

models_list = ""
for i in range(len(models)):
    models_list += f"\n     [{i}] - {models[i]}"

user_choice: str = models[int(
    input("What model would you like to play :" + models_list + '\n').strip())]
model_path = path.normpath(path.join(
    getcwd(), 'models', user_choice))

IS_AIM = user_choice.startswith(
    'model_aim')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MouseNet().to(device) if IS_AIM else ClicksNet().to(device)
model.load_state_dict(torch.load(
    model_path
)['state'])
model.eval()
print("Model Loaded, 'Shitf + R' To Toggle the model")

IMAGE_BUFF = Queue()

elapsed = 1


def toggle_capture():
    global do_prediction
    do_prediction = not do_prediction
    if do_prediction:
        IMAGE_BUFF.queue.clear()


keyboard.add_hotkey('shift+r', callback=toggle_capture)


# def handle_video():
#     global elapsed
#     latest_image = None
#     while True:
#         # print("FPS:", 1/elapsed, end='\r')
#         frame = IMAGE_BUFF.get()
#         if frame is not None:
#             cv2.imshow("Ai Vision", frame)
#             cv2.waitKey(1)

#     cv2.destroyAllWindows()


# Thread(daemon=True, group=None, target=handle_video).start()
STATE_DISPLAY = {
    0: "Idle    ",
    1: "Button 1",
    2: "Button 2"
}

try:
    while True:
        start = time.time()
        frame = cap.capture(*PLAY_AREA_CAPTURE_PARAMS)
        if frame is not None:
            cv_img = cv2.cvtColor(cv2.resize(
                frame,  (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
                    PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)

            elapsed = time.time() - start
            stacked = np.stack([cv_img, cv_img, cv_img], axis=-1)
            if do_prediction:
                converted_frame = transform(stacked)

                inputs = converted_frame.reshape(
                    (1, converted_frame.shape[0], converted_frame.shape[1], converted_frame.shape[2])).type(torch.FloatTensor).to(device)

                output = model(inputs)

                if IS_AIM:
                    mouse_x_percent, mouse_y_percent = output[0]
                    position = (PLAY_AREA_CAPTURE_PARAMS[2] + int(mouse_x_percent * PLAY_AREA_CAPTURE_PARAMS[0]), PLAY_AREA_CAPTURE_PARAMS[3] + int(
                        mouse_y_percent * PLAY_AREA_CAPTURE_PARAMS[1]))
                    win32api.SetCursorPos(position)
                    print(position, end='\r')
                else:
                    _, predicated = torch.max(output, dim=1)

                    probs = torch.softmax(output, dim=1)
                    prob = probs[0][predicated.item()]
                    print(
                        f"Decision {STATE_DISPLAY[predicated.item()]} :: chance {prob.item():.2f}", end='\r')
                    if prob.item() > 0:  # 0.7:
                        set_key_state(predicated.item())

            # IMAGE_BUFF.put(stacked)
            # elapsed = time.time() - start
            # wait_time = 0.01 - elapsed
            # if wait_time > 0:
            #     time.sleep(wait_time)


except KeyboardInterrupt as e:
    pass
