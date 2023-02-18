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
        self.stacks = 5
        self.agent = OsuAgent(self.stacks)
        self.memory = deque([], maxlen=CAPACITY_MAX)
        self.lr = 1e-4
        self.gamma = 0.99
        self.tau = 1.0
        self.model = DQN(stacks=self.stacks).to(PYTORCH_DEVICE)
        self.target = DQN(stacks=self.stacks).to(PYTORCH_DEVICE)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.epsilon = 0
        self.plays = 0

    def remember(self, mem):
        self.memory.append(mem)

    def step(self):

        prev_state = self.agent.get_state()

        if prev_state is None:
            return True

        last_playfield, last_score, last_accuracy, last_game_time = prev_state

        action = self.sample(last_playfield)

        self.agent.do_action(action)

        time.sleep(0.05)

        new_state = self.agent.get_state()

        if new_state is None:
            return True

        playfield, score, accuracy, game_time = new_state

        reward = 0

        if accuracy < last_accuracy:
            if action == 0:
                reward = -800
            else:
                reward = 400

        is_over = accuracy < 5 and game_time > 2

        self.remember(
            np.array([last_playfield, action, reward, playfield, int(is_over)], dtype=object))

        is_over = is_over if not self.train() else True
        return is_over

    def reset(self):
        self.plays += 1
        self.agent.reset()
        print("Attempt", self.plays)

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
        if random.randint(0, 200) > self.epsilon:
            return random.randint(0, 1)
        else:
            with torch.no_grad():
                return int(self.predict_one(state).item())

    def train(self):
        if len(self.memory) < CAPACITY_MAX:
            return False

        states, actions, rewards, next_states, dones = zip(
            *self.memory)

        state: torch.Tensor = torch.from_numpy(
            np.stack(states, axis=0)).type(torch.FloatTensor).to(PYTORCH_DEVICE)
        next_state: torch.Tensor = torch.from_numpy(
            np.stack(next_states, axis=0)).type(torch.FloatTensor).to(PYTORCH_DEVICE)
        reward: torch.Tensor = torch.from_numpy(
            np.stack(rewards, axis=0)).type(torch.LongTensor).to(PYTORCH_DEVICE)
        done: torch.Tensor = torch.from_numpy(
            np.stack(dones, axis=0)).to(PYTORCH_DEVICE)

        action = torch.from_numpy(
            np.stack(actions, axis=0)).type(torch.LongTensor).to(PYTORCH_DEVICE)

        with torch.no_grad():
            target_max, _ = self.target(next_state).max(dim=1)
            td_target = reward.flatten() + self.gamma * target_max * (1 - done.flatten())

        old_val = self.model(state).squeeze(1)[action]

        loss = self.criterion(td_target, old_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Optimized with loss {loss.item()}")

        for target_network_param, q_network_param in zip(self.target.parameters(), self.model.parameters()):
            target_network_param.data.copy_(
                self.tau * q_network_param.data +
                (1.0 - self.tau) * target_network_param.data
            )

        self.memory.clear()
        self.epsilon += 1
        return True
