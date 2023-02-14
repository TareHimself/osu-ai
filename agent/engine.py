import gym
import numpy as np
from gym import Env
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT
from agent.interface import OsuGymInterface


def run():
    my_config = DEFAULT_CONFIG_DICT
    my_config['interface'] = OsuGymInterface
    my_config["time_step_duration"] = 0.05
    my_config["start_obs_capture"] = 0.05
    my_config["time_step_timeout_factor"] = 1.0
    my_config["ep_max_length"] = np.inf
    my_config["act_buf_len"] = 4
    my_config["reset_act_buf"] = True

    env: Env = gym.make("real-time-gym-v0", my_config)

    obs, info = env.reset()

    while not (terminated or truncated):
    act = model(obs)
    obs, rew, terminated, truncated, info = env.step(act)
    print(f"rew:{rew}")

    while True:  # when this loop is broken, the current time-step will timeout
    action = (obs)  # inference takes a random amount of time
    obs, rew, terminated, truncated, info = env.step(act)  # transparently adapts to this duration
