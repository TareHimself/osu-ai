import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from constants import PYTORCH_DEVICE
from rl.memory import ReplayMemory


class DQN(torch.nn.Module):
    """
    Works so far

    Args:
        torch (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.conv = resnet18(weights=None)
        num_ftrs = self.conv.fc.in_features
        self.conv.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, images):
        return self.conv(images)

    def save_model(self, path="test.pt"):
        torch.save(self.state_dict(), path)

    def load_model(self, path="test.pt"):
        self.load_state_dict(torch.load(path))
