import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")

# works so far


class ClicksNet(torch.nn.Module):
    """
    Works so far

    Args:
        torch (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        # resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.conv.fc.in_features
        self.conv.fc = nn.Linear(num_ftrs, 3)
        # self.conv.fc = nn.Sequential(
        #     nn.Linear(num_ftrs, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 3)
        # )

    def forward(self, images):
        return self.conv(images)


class MouseNet(torch.nn.Module):
    """
    Untested

    Args:
        torch (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        # resnet18()
        self.conv = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.conv.fc.in_features
        print(num_ftrs)
        self.conv.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, images):
        return self.conv(images)
