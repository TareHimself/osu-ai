import torch
import torch.nn as nn


class DQN(torch.nn.Module):
    """
    Works so far

    Args:
        torch (_type_): _description_
    """

    def __init__(self, action_space=1, stacks=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(stacks, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7488, 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
        )

    def forward(self, images):
        return self.conv(images)

    def save_model(self, path="test.pt"):
        torch.save(self.state_dict(), path)

    def load_model(self, path="test.pt"):
        self.load_state_dict(torch.load(path))
