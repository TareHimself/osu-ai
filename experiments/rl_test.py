import time
import random
import gymnasium
from rl.agent import OsuAgent
from rl.env import OsuEnviroment

env = OsuEnviroment()
state = env.reset()
try:
    while True:
        done_with_episode = False
        while not done_with_episode:
            done_with_episode = env.step()
            time.sleep(0.1)
        env.reset()
except KeyboardInterrupt:
    env.agent.kill()
