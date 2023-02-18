import torch
import keyboard
import win32api
from threading import Thread
from torch.nn import Module
from torch import Tensor
from ai.models import ActionsNet, AimNet, OsuAiModel
from constants import FINAL_RESIZE_PERCENT, PLAY_AREA_CAPTURE_PARAMS, PYTORCH_DEVICE
from windows import WindowCapture
from collections import deque
import cv2


class EvalThread(Thread):

    def __init__(self, model_path: str, game_window_name: str = "osu! (development)", eval_key: str = '\\', stack_num=3):
        super().__init__(group=None, daemon=True)
        self.game_window_name = game_window_name
        self.model_path = model_path
        self.eval_key = eval_key
        self.stack_num = stack_num
        self.eval = True
        self.start()

    def load_model(self, path: str) -> OsuAiModel:
        pass

    def on_output(self, output: Tensor):
        pass

    def on_eval_ready(self):
        pass

    def kill(self):
        self.eval = False

    def run(self):
        model = self.load_model(self.model_path)
        frame_buffer = deque(maxlen=self.stack_num)
        eval_this_frame = False

        def toggle_eval():
            nonlocal eval_this_frame
            eval_this_frame = not eval_this_frame

        keyboard.add_hotkey(self.eval_key, callback=toggle_eval)

        cap = WindowCapture(self.game_window_name)
        self.on_eval_ready()
        with torch.no_grad():
            while self.eval:
                frame, stacked = cap.capture(
                    list(frame_buffer), self.stack_num, (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
                        PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)), *PLAY_AREA_CAPTURE_PARAMS)

                frame_buffer.append(frame)

                if eval_this_frame:
                    cv2.imshow("Debug", stacked.transpose(1, 2, 0))
                    cv2.waitKey(10)

                    converted_frame = torch.from_numpy(stacked / 255).type(
                        torch.FloatTensor).to(PYTORCH_DEVICE)

                    inputs = converted_frame.reshape(
                        (1, converted_frame.shape[0], converted_frame.shape[1], converted_frame.shape[2]))

                    out = model(inputs)
                    self.on_output(out)

        del cap


class ActionsThread(EvalThread):
    KEYS_STATE_TO_STRING = {
        0: "Idle    ",
        1: "Button 1",
        2: "Button 2"
    }

    def __init__(self,  model_path: str, game_window_name: str = "osu! (development)", eval_key: str = '\\', stack_num=3):
        super().__init__(model_path, game_window_name, eval_key, stack_num)

    def load_model(self, path: str) -> OsuAiModel:
        loaded = ActionsNet().to(PYTORCH_DEVICE)
        loaded.load(path)
        return loaded

    def on_eval_ready(self):
        print(f"Actions Model Ready,Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        _, predicated = torch.max(output, dim=1)
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicated.item()]
        print("State", predicated.item())
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
    def __init__(self,  model_path: str, game_window_name: str = "osu! (development)", eval_key: str = '\\', stack_num=3):
        super().__init__(model_path, game_window_name, eval_key, stack_num)

    def load_model(self, path: str) -> OsuAiModel:
        loaded = AimNet().to(PYTORCH_DEVICE)
        loaded.load(path)
        return loaded

    def on_eval_ready(self):
        print(f"Aim Model Ready,Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        mouse_x_percent, mouse_y_percent = output[0]
        position = (int((mouse_x_percent * PLAY_AREA_CAPTURE_PARAMS[0]) + PLAY_AREA_CAPTURE_PARAMS[2]), int(
            (mouse_y_percent * PLAY_AREA_CAPTURE_PARAMS[1]) + PLAY_AREA_CAPTURE_PARAMS[3]))
        win32api.SetCursorPos(position)
