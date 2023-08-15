import os.path
import time
import cv2
import numpy as np
import torch
import keyboard
import win32api
from threading import Thread
from torch import Tensor
from ai.models import ActionsNet, AimNet, OsuAiModel, CombinedNet
from ai.constants import FINAL_PLAY_AREA_SIZE, FRAME_DELAY, PYTORCH_DEVICE, MODELS_DIR
from ai.utils import FixedRuntime, derive_capture_params
from collections import deque
from mss import mss
from ai.enums import EPlayAreaIndices

# 'osu!'  #
DEFAULT_OSU_WINDOW = 'osu!'  # "osu! (development)"


class EvalThread(Thread):

    def __init__(self, model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__(group=None, daemon=True)
        self.game_window_name = game_window_name
        self.model_id = model_id
        self.capture_params = derive_capture_params()
        self.eval_key = eval_key
        self.eval = True
        self.start()


    def get_model(self):
        model = torch.jit.load(os.path.join(MODELS_DIR, self.model_id, 'model.pt'))
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, self.model_id, 'weights.pt')))
        model.to(PYTORCH_DEVICE)
        model.eval()
        return model

    def on_output(self, output: Tensor):
        pass

    def on_eval_ready(self):
        print("Unknown Model Ready")

    def kill(self):
        self.eval = False

    @torch.no_grad()
    def run(self):
        eval_model = self.get_model()
        with torch.inference_mode():
            frame_buffer = deque(maxlen=eval_model.channels)
            eval_this_frame = False

            def toggle_eval():
                nonlocal eval_this_frame
                eval_this_frame = not eval_this_frame

            keyboard.add_hotkey(self.eval_key, callback=toggle_eval)

            self.on_eval_ready()

            print(self.capture_params)
            with mss() as sct:
                monitor = {"top": self.capture_params[EPlayAreaIndices.OffsetY.value],
                           "left": self.capture_params[EPlayAreaIndices.OffsetX.value],
                           "width": self.capture_params[EPlayAreaIndices.Width.value],
                           "height": self.capture_params[EPlayAreaIndices.Height.value]}

                while self.eval:
                    with FixedRuntime(target_time=FRAME_DELAY):  # limit capture to every "FRAME_DELAY" seconds
                        if eval_this_frame:
                            frame = np.array(sct.grab(monitor))
                            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), FINAL_PLAY_AREA_SIZE)

                            needed = eval_model.channels - len(frame_buffer)

                            if needed > 0:
                                for i in range(needed):
                                    frame_buffer.append(frame)
                            else:
                                frame_buffer.append(frame)

                            stacked = np.stack(frame_buffer)

                            frame_buffer.append(frame)
                            cv2.imshow("Debug", stacked[0:3].transpose(1, 2, 0))
                            cv2.waitKey(1)

                            converted_frame = torch.from_numpy(stacked / 255).type(
                                torch.FloatTensor).to(PYTORCH_DEVICE)

                            inputs = converted_frame.reshape(
                                (1, converted_frame.shape[0], converted_frame.shape[1], converted_frame.shape[2]))

                            out: torch.Tensor = eval_model(inputs)

                            self.on_output(out.detach())

            keyboard.remove_hotkey(toggle_eval)


class ActionsThread(EvalThread):
    KEYS_STATE_TO_STRING = {
        0: "Idle    ",
        1: "Button 1",
        2: "Button 2"
    }

    def __init__(self, model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__(model_id, game_window_name, eval_key)


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
    def __init__(self, model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__(model_id, game_window_name, eval_key)

    # def get_model(self):
    #     # model = torch.jit.load(os.path.join(MODELS_DIR, self.model_id, 'model.pt'))
    #     # model.load_state_dict(torch.load(os.path.join(MODELS_DIR, self.model_id, 'weights.pt')))
    #     model = AimNet.load(self.model_id)
    #     model.to(PYTORCH_DEVICE)
    #     model.eval()
    #     return model
    
    def on_eval_ready(self):
        print(f"Aim Model Ready,Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        mouse_x_percent, mouse_y_percent = output[0]
        position = (int((mouse_x_percent * self.capture_params[EPlayAreaIndices.Width.value]) + self.capture_params[
            EPlayAreaIndices.OffsetX.value]), int(
            (mouse_y_percent * self.capture_params[EPlayAreaIndices.Height.value]) + self.capture_params[
                EPlayAreaIndices.OffsetY.value]))
        win32api.SetCursorPos(position)


class CombinedThread(EvalThread):
    def __init__(self, model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__(model_id, game_window_name, eval_key)

    def on_eval_ready(self):
        print(f"Combined Model Ready,Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        mouse_x_percent, mouse_y_percent, k1_prob, k2_prob = output[0]
        position = (int((mouse_x_percent * self.capture_params[EPlayAreaIndices.Width.value]) + self.capture_params[
            EPlayAreaIndices.OffsetX.value]), int(
            (mouse_y_percent * self.capture_params[EPlayAreaIndices.Height.value]) + self.capture_params[
                EPlayAreaIndices.OffsetY.value]))
        win32api.SetCursorPos(position)

        if k1_prob >= 0.5:
            keyboard.press('z')
        else:
            keyboard.release('z')

        if k2_prob >= 0.5:
            keyboard.press('x')
        else:
            keyboard.release('x')
