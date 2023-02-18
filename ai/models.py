import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from constants import PYTORCH_DEVICE


class OsuAiModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def save(self, path: str, data=None):
        if data is None:
            torch.save({
                'state': self.state_dict()
            }, path)
        else:
            torch.save(data, path)

    def load(self, path: str):
        saved = torch.load(path)
        self.load_state_dict(saved['state'])


class AimNet(OsuAiModel):
    """
    Works

    Args:
        torch (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        # resnet18()
        self.conv = resnet18(weights=None)
        num_ftrs = self.conv.fc.in_features
        self.conv.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, images):
        return self.conv(images)


class ActionsNet(OsuAiModel):
    """
    Works so far

    Args:
        torch (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.conv = resnet18(weights=None)
        num_ftrs = self.conv.fc.in_features
        self.conv.fc = nn.Linear(num_ftrs, 3)

    def forward(self, images):
        return self.conv(images)
