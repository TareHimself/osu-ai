import keyboard
import win32api

from constants import PLAY_AREA_CAPTURE_PARAMS


class OsuPlayer:
    def __init__():
        pass

    def do_action(self, mouse_x_percent, mouse_y_percent, key_state):

        mouse_x_scaled = PLAY_AREA_CAPTURE_PARAMS[2] + \
            (mouse_x_percent * PLAY_AREA_CAPTURE_PARAMS[0])
        mouse_y_scaled = PLAY_AREA_CAPTURE_PARAMS[3] + \
            (mouse_y_percent * PLAY_AREA_CAPTURE_PARAMS[1])

        win32api.SetCursorPos((mouse_x_scaled,  mouse_y_scaled))
        if key_state >= 0.5:
            keyboard.press('z')
        else:
            keyboard.release('z')
