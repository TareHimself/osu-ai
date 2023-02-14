import keyboard
import numpy as np
import win32api
import time
import traceback
import os
from threading import Thread
from agent.osu_player import OsuPlayer
from constants import FINAL_RESIZE_PERCENT, PLAY_AREA_CAPTURE_PARAMS
from rtgym.envs import RealTimeGymInterface
from gym import spaces


class OsuGymInterface:
    def __init__(self) -> None:
        super().__init__(self)
        self.osu = OsuPlayer()
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        """
    Implement this class for your application
    """

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        If control is None, does nothing
        Args:
            control: np.array of the dimension of the action-space
        """
        if control is not None:
            self.osu.do_action(control[0], control[1])

    def wait(self):
        self.send_control(self.get_default_action())

    def reset(self):
        """
        Returns:
            obs: must be a list
        Note: Do NOT put the action buffer in the returned obs (automated)
        """
        screen = self.osu.get_game()
        self.osu.accuracy = 1
        obs = [screen]
        return obs

    def get_obs_rew_terminated_info(self):
        """Returns observation, reward, terminated and info from the device.

        Note:

        Returns:
            obs: list (corresponding to the tuple from get_observation_space)
            rew: scalar
            terminated: bool
            info: dict

        Note: Do NOT put the action buffer in obs (automated).
        """
        # return obs, rew, terminated, info

        screen = self.osu.get_game()
        acc = self.osu.get_accuracy()
        obs = [screen]
        rew = acc - 1
        terminated = acc < 0.8
        info = {}
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        Returns:
            observation_space: gym.spaces.Tuple
        Note: Do NOT put the action buffer here (automated)
        """

        image_space = spaces.Box(
            low=0, high=255, shape=(int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
                PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT), 3), dtype=np.uint8)
        # return spaces.Tuple(...)

        return spaces.Tuple((image_space))

    def get_action_space(self):
        """
        Returns:
            action_space: gym.spaces.Box
        """
        return spaces.Box(low=0, high=1.0, shape=(2,))

    def get_default_action(self):
        """
        Returns:
            default_action: numpy array of the dimension of the action space
        initial action at episode start
        """

        return np.array([0.0, 0.0], dtype='float32')
