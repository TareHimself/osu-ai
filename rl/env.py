import gymnasium
import time
import torch
import numpy as np
import random
from torch import optim, nn
from gymnasium import spaces
from constants import PLAY_AREA_CAPTURE_PARAMS, FINAL_RESIZE_PERCENT, PYTORCH_DEVICE
from rl.agent import OsuAgent
from collections import deque
from rl.dqn import DQN

IMAGE_SHAPE = (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT),
               int(PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT), 3)

CAPACITY_MAX = 32


class OsuEnviroment():

    def __init__(self) -> None:
        super().__init__()
        self.agent = OsuAgent()
        self.memory = deque([], maxlen=CAPACITY_MAX)
        self.lr = 0.001
        self.gamma = 0.7
        self.model = DQN().to(PYTORCH_DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.epsilon = 0

    def remember(self, mem):
        self.memory.append(mem)

    def step(self):

        last_playfield, last_score, last_accuracy, last_game_time = self.agent.get_state()

        action = self.sample(last_playfield)

        self.agent.do_action(action)

        time.sleep(0.05)

        playfield, score, accuracy, game_time = self.agent.get_state()

        reward = 0

        if score > last_score:
            reward = 100
        elif accuracy < last_accuracy:
            score = -200

        is_over = accuracy < 5 and game_time > 10

        self.remember(
            np.array([last_playfield, action, reward, playfield, is_over]))

        is_over = is_over if not self.train() else True

        return is_over

    def reset(self):
        self.epsilon += 1
        self.agent.reset()

    def predict_one(self, img):
        model_in = torch.from_numpy(img)

        return self.predict_one_tensor(model_in)

    def predict_one_tensor(self, model_in):
        model_in = model_in.reshape(
            (1, model_in.shape[0], model_in.shape[1], model_in.shape[2])).type(
            torch.FloatTensor).to(PYTORCH_DEVICE)

        output = self.model(model_in)

        _, predicated = torch.max(output, dim=1)
        return predicated

    def sample(self, state):
        if random.randint(0, 100) > self.epsilon:
            return random.randint(0, 1)
        else:
            with torch.no_grad():
                return int(self.predict_one(state).item())

    def train(self):
        if len(self.memory) < CAPACITY_MAX:
            return False

        states, actions, rewards, next_states, dones = zip(
            *self.memory)

        state = torch.from_numpy(
            np.stack(states, axis=0)).type(torch.FloatTensor).to(PYTORCH_DEVICE)
        next_state = torch.from_numpy(
            np.stack(next_states, axis=0)).type(torch.FloatTensor).to(PYTORCH_DEVICE)
        reward = torch.from_numpy(
            np.stack(rewards, axis=0)).type(torch.LongTensor).to(PYTORCH_DEVICE)
        done = torch.from_numpy(
            np.stack(dones, axis=0)).to(PYTORCH_DEVICE)

        batch_index = np.arange(CAPACITY_MAX, dtype=np.int32)

        action = torch.from_numpy(
            np.stack(actions, axis=0)).type(torch.LongTensor)

        q_eval: torch.Tensor = self.model(state)[batch_index, action]
        q_next: torch.Tensor = self.model(next_state)
        q_next[done] = 0.0

        q_target = reward + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.criterion(q_target, q_eval).to(PYTORCH_DEVICE)

        loss.backward()

        self.optimizer.step()

        print(f"Optimized with loss {loss.item()}")
        self.memory.clear()
        return True
