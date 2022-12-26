import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from constants import PYTORCH_DEVICE

# works so far


# class ActionsNet(torch.nn.Module):
#     """
#     Works so far

#     Args:
#         torch (_type_): _description_
#     """

#     def __init__(self):
#         super().__init__()
#         # resnet18(weights=ResNet18_Weights.DEFAULT)
#         self.conv = resnet18(weights=ResNet18_Weights.DEFAULT)
#         num_ftrs = self.conv.fc.in_features
#         self.conv.fc = nn.Linear(num_ftrs, 3)
#         # self.conv.fc = nn.Sequential(
#         #     nn.Linear(num_ftrs, 128),
#         #     nn.ReLU(),
#         #     nn.Linear(128, 3)
#         # )

#     def forward(self, images):
#         return self.conv(images)


# class ActionsNet(nn.Module):
#     def __init__(self, hidden_dim=128, n_layers=3, dropout=0.2):
#         super().__init__()

#         # Load pre-trained ResNet-18 model and freeze its weights
#         self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
#         for param in self.resnet.parameters():
#             param.requires_grad = False

#         # Replace the fully-connected layer with a linear layer
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_ftrs, hidden_dim)

#         # Add an LSTM layer
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim,
#                             num_layers=n_layers, dropout=dropout)

#         # Add a linear layer for classification
#         self.fc = nn.Linear(hidden_dim, 3)

#     def forward(self, x):
#         # Extract features from the input image using the ResNet-18 model
#         x = self.resnet(x)
#         # Pass the extracted features through the LSTM layer
#         x, (hidden, cell) = self.lstm(x.unsqueeze(0))
#         # Pass the final hidden state through the classification layer
#         x = self.fc(hidden[-1])
#         return x


class ActionsNet(torch.nn.Module):
    """
    Works so far

    Args:
        torch (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.conv = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.conv.fc.in_features
        self.conv.fc = nn.Linear(num_ftrs, 3)

    def forward(self, images):
        return self.conv(images)


# class ActionsNet(torch.nn.Module):
#     """
#     Works so far

#     Args:
#         torch (_type_): _description_
#     """

#     def __init__(self):
#         super().__init__()
#         # resnet18(weights=ResNet18_Weights.DEFAULT)
#         self.conv = resnet18(weights=ResNet18_Weights.DEFAULT)
#         num_ftrs = self.conv.fc.in_features
#         self.hidden_size = 128
#         self.num_layers = 2
#         self.conv.fc = nn.LSTM(num_ftrs,
#                                self.hidden_size, self.num_layers, batch_first=True)
#         self.fc = nn.Linear(self.hidden_size, 3)

#     def forward(self, images):
#         x, _ = self.conv(images)

#         # print(x.shape)
#         # # Forward propagate LSTM
#         # # out: tensor of shape (batch_size, seq_length, hidden_size)
#         # out, _ = self.lstm(x)

#         return self.fc(x)


class AimNet(torch.nn.Module):
    """
    Untested

    Args:
        torch (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        # resnet18()
        self.conv = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.conv.fc.in_features
        self.conv.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, images):
        return self.conv(images)

# class AimNet(torch.nn.Module):
#     """
#     Untested

#     Args:
#         torch (_type_): _description_
#     """

#     def __init__(self):
#         super().__init__()
#         # resnet18()
#         self.conv = resnet18(weights=ResNet18_Weights.DEFAULT)
#         num_ftrs = self.conv.fc.in_features
#         self.conv.fc = nn.Sequential(
#             nn.Linear(num_ftrs, num_ftrs),
#             nn.ReLU(),
#             nn.Linear(num_ftrs, num_ftrs),
#             nn.ReLU(),
#             nn.Linear(num_ftrs, 1000),
#             nn.ReLU(),
#             nn.Linear(1000, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 2)
#         )

#     def forward(self, images):
#         return self.conv(images)


# class AimNet0(torch.nn.Module):
#     """
#     Untested

#     Args:
#         torch (_type_): _description_
#     """

#     def __init__(self):
#         super().__init__()
#         # resnet18()
#         self.conv = resnet50()
#         num_ftrs = self.conv.fc.in_features
#         self.conv.fc = nn.Sequential(
#             nn.Linear(num_ftrs, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2),
#         )

#     def forward(self, images):
#         return self.conv(images)
