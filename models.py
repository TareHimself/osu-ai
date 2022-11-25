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
        # resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv = resnet18()
        num_ftrs = self.conv.fc.in_features
        self.conv.fc = nn.Sequential(nn.Linear(
            num_ftrs, 2), nn.Sigmoid())

    def forward(self, images):
        return self.conv(images)
