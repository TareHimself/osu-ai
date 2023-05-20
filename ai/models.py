import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from constants import CURRENT_STACK_NUM,  FINAL_PLAY_AREA_SIZE


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
        print("FEATURES", num_ftrs)
        self.conv.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
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


class TestModel(OsuAiModel):
    def __init__(self, input_channels=CURRENT_STACK_NUM, height=FINAL_PLAY_AREA_SIZE[1], width=FINAL_PLAY_AREA_SIZE[0]):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.3)

        self.fc1 = nn.Linear(128 * (height//8) * (width//8), 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        return x
