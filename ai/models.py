import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from constants import CURRENT_STACK_NUM, PYTORCH_DEVICE


def res18WithChannels(resnet, channels=4):

    new_in_channels = 4

    model = resnet

    layer = model.conv1

    # Creating new Conv2d layer
    new_layer = nn.Conv2d(in_channels=new_in_channels,
                          out_channels=layer.out_channels,
                          kernel_size=layer.kernel_size,
                          stride=layer.stride,
                          padding=layer.padding,
                          bias=layer.bias)

    # Here will initialize the weights from new channel with the red channel weights
    copy_weights = 0

    # Copying the weights from the old to the new layer
    new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

    # Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
    for i in range(new_in_channels - layer.in_channels):
        channel = layer.in_channels + i
        new_layer.weight[:, channel:channel+1, :, :] = layer.weight[:,
                                                                    copy_weights:copy_weights+1, ::].clone()
    new_layer.weight = nn.Parameter(new_layer.weight)

    model.conv1 = new_layer
    return model


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
        self.conv.conv1 = nn.Conv2d(
            CURRENT_STACK_NUM, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.conv.fc.in_features
        print("FEATURES",num_ftrs)
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
        self.conv.conv1 = nn.Conv2d(
            CURRENT_STACK_NUM, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.conv.fc.in_features
        self.conv.fc = nn.Linear(num_ftrs, 3)

    def forward(self, images):
        return self.conv(images)
