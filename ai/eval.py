import time
import cv2
import numpy as np
import torch
import keyboard
import win32api
from threading import Thread
from torch.nn import Module
from torch import Tensor
from ai.models import ActionsNet, AimNet, OsuAiModel
from constants import FINAL_RESIZE_PERCENT, FRAME_DELAY, PLAY_AREA_CAPTURE_PARAMS, PYTORCH_DEVICE,PLAY_AREA_INDICES
from utils import FixedRuntime
from collections import deque
from mss import mss
#'osu!'  #
DEFAULT_OSU_WINDOW = 'osu!'  #"osu! (development)"


class EvalThread(Thread):

    def __init__(self, model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__(group=None, daemon=True)
        self.game_window_name = game_window_name
        self.model_id = model_id
        self.eval_key = eval_key
        self.eval = True
        self.start()

    def get_model(self, model_id: str) -> OsuAiModel:
        pass

    def on_output(self, output: Tensor):
        pass

    def on_eval_ready(self):
        print("Unknown Model Ready")

    def kill(self):
        self.eval = False

    @torch.no_grad()
    def run(self):
        eval_model = self.get_model(self.model_id).to(PYTORCH_DEVICE)
        eval_model.eval()
        with torch.inference_mode():
            frame_buffer = deque(maxlen=eval_model.channels)
            eval_this_frame = False

            def toggle_eval():
                nonlocal eval_this_frame
                eval_this_frame = not eval_this_frame

            keyboard.add_hotkey(self.eval_key, callback=toggle_eval)

            self.on_eval_ready()
            with torch.inference_mode():
                with mss() as sct:
                    monitor = monitor = {"top": PLAY_AREA_CAPTURE_PARAMS[PLAY_AREA_INDICES.Y_OFFSET], "left": PLAY_AREA_CAPTURE_PARAMS[PLAY_AREA_INDICES.X_OFFSET], "width": PLAY_AREA_CAPTURE_PARAMS[PLAY_AREA_INDICES.WIDTH], "height": PLAY_AREA_CAPTURE_PARAMS[PLAY_AREA_INDICES.HEIGHT]}
                    
                    while self.eval:
                        with FixedRuntime(target_time=FRAME_DELAY): # limit capture to every "FRAME_DELAY" seconds
                            if eval_this_frame:
                                frame = np.array(sct.grab(monitor))
                                frame = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),(int(PLAY_AREA_CAPTURE_PARAMS[PLAY_AREA_INDICES.WIDTH] * FINAL_RESIZE_PERCENT), int(
                                        PLAY_AREA_CAPTURE_PARAMS[PLAY_AREA_INDICES.HEIGHT] * FINAL_RESIZE_PERCENT)))

                                needed = eval_model.channels - len(frame_buffer)

                                if needed > 0:
                                    for i in range(needed):
                                        frame_buffer.append(frame)
                                else:
                                    frame_buffer.append(frame)

                                stacked = np.stack(frame_buffer)

                                start = time.time()
                                frame_buffer.append(frame)
                                # cv2.imshow("Debug", stacked.transpose(1, 2, 0))
                                # cv2.waitKey(1)

                                converted_frame = torch.from_numpy(stacked / 255).type(
                                    torch.FloatTensor).to(PYTORCH_DEVICE)

                                inputs = converted_frame.reshape(
                                    (1, converted_frame.shape[0], converted_frame.shape[1], converted_frame.shape[2]))

                                out: torch.Tensor = eval_model(inputs)
                                self.on_output(out.detach())
                                end = time.time() - start
                                # print(f"Delay {end}") # debug capture speed

            keyboard.remove_hotkey(toggle_eval)


class ActionsThread(EvalThread):
    KEYS_STATE_TO_STRING = {
        0: "Idle    ",
        1: "Button 1",
        2: "Button 2"
    }

    def __init__(self,  model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__(model_id, game_window_name, eval_key)

    def get_model(self, model_id: str) -> OsuAiModel:
        return ActionsNet.load(model_id)

    def on_eval_ready(self):
        print(f"Actions Model Ready,Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        _, predicated = torch.max(output, dim=1)
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicated.item()]
        if prob.item() > 0:  # 0.7:
            state = predicated.item()
            if state == 0:
                keyboard.release('x')
                keyboard.release('z')
            elif state == 1:
                keyboard.release('z')
                keyboard.press('x')
            elif state == 2:
                keyboard.release('x')
                keyboard.press('z')


class AimThread(EvalThread):
    def __init__(self,  model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__(model_id, game_window_name, eval_key)


    def get_model(self, model_id: str) -> OsuAiModel:
        return AimNet.load(model_id)

    def on_eval_ready(self):
        print(f"Aim Model Ready,Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        mouse_x_percent, mouse_y_percent = output[0]
        position = (int((mouse_x_percent * PLAY_AREA_CAPTURE_PARAMS[0]) + PLAY_AREA_CAPTURE_PARAMS[2]), int(
            (mouse_y_percent * PLAY_AREA_CAPTURE_PARAMS[1]) + PLAY_AREA_CAPTURE_PARAMS[3]))
        win32api.SetCursorPos(position)
