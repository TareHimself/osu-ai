from gym import spaces
import keyboard
import numpy as np
from rtgym.envs import RealTimeGymInterface
import win32api
from agent.osu_player import OsuPlayer
from constants import FINAL_RESIZE_PERCENT, PLAY_AREA_CAPTURE_PARAMS


class OsuGymInterface:
    def __init__(self) -> None:
        super().__init__(self)
        self.osu = OsuPlayer()
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
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
            self.osu.do_action(control[0], control[1], control[2])

    def wait(self):
        self.send_control(self.get_default_action())

    def reset(self):
        """
        Returns:
            obs: must be a list
        Note: Do NOT put the action buffer in the returned obs (automated)
        """
        # return obs

        raise NotImplementedError

    def get_obs_rew_done(self):
        """
        Returns:
            obs: list
            rew: scalar
            done: boolean
        Note: Do NOT put the action buffer in obs (automated)
        """
        # return obs, rew, done

        raise NotImplementedError

    def get_observation_space(self):
        """
        Returns:
            observation_space: gym.spaces.Tuple
        Note: Do NOT put the action buffer here (automated)
        """
        # return spaces.Tuple(...)

        raise NotImplementedError

    def get_action_space(self):
        """
        Returns:
            action_space: gym.spaces.Box
        """
        return spaces.Box(low=0, high=1.0, shape=(3,))

    def get_default_action(self):
        """
        Returns:
            default_action: numpy array of the dimension of the action space
        initial action at episode start
        """

        return np.array([0.0, 0.0, 0.0], dtype='float32')
