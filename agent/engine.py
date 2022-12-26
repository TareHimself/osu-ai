# import gym
# from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT
# from agent.interface import OsuGymInterface


# def run():
#     gym_config = DEFAULT_CONFIG_DICT
#     gym_config['interface'] = OsuGymInterface

#     gym_enviroment = gym.make(
#         "real-time-gym-v0", gym_config, disable_env_checker=True)

#     obs, info = gym_enviroment.reset()


#     while True: # when this loop is broken, the current time-step will timeout
#     action = (obs)  # inference takes a random amount of time
#     obs, rew, terminated, truncated, info = env.step(
#         act)  # transparently adapts to this duration
