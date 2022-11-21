from os import path, mkdir, getcwd
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
from models import ClicksNet
transform = transforms.ToTensor()

cap = WindowCapture("osu!")


capture_frame = False


def toggle_capture():
    global capture_frame
    capture_frame = not capture_frame


keyboard.add_hotkey('shift+r', callback=toggle_capture)

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


FINAL_RESIZE_PERCENT = 0.3

CAPTURE_PARAMS = derive_capture_params()
RESIZED_PARAMS = derive_capture_params()

model_path = path.normpath(path.join(
    getcwd(), 'models', f'{input("What model would you like to play ? ").strip().lower()}.pt'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ClicksNet().to(device)
model.load_state_dict(torch.load(
    model_path
)['state'])
model.eval()
print("Model Loaded, 'Shitf + R' To Toggle the model")

# IMAGE_BUFF = Queue()

# elapsed = 1


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

try:
    while True:
        start = time.time()
        frame = cap.capture(*CAPTURE_PARAMS)
        if frame is not None and capture_frame:
            cv_img = cv2.resize(
                frame,  (int(CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
                    CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)), interpolation=cv2.INTER_LINEAR)

            elapsed = time.time() - start
            inputs = transform(np.array(cv_img)).reshape(
                (1, 3, 300, 399)).to(device)
            output = model(inputs)

            _, predicated = torch.max(output, dim=1)

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicated.item()]
            print(
                f"Decision {predicated.item()} :: chance {prob.item()}", end='\r')
            if prob.item() > 0:  # 0.7:
                set_key_state(predicated.item())

            # IMAGE_BUFF.put(cv_img)
            # elapsed = time.time() - start
            # wait_time = 0.01 - elapsed
            # if wait_time > 0:
            #     time.sleep(wait_time)


except KeyboardInterrupt as e:
    pass
