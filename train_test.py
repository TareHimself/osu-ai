import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dataset import OsuDataset
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet18
trans = transforms.ToTensor()

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def numpyToTensor(arr, d):
    print(arr.shape)
    return trans(arr)
    print(arr.shape)
    Tn = torch.Tensor(d[0], d[1])
    Ts = torch.stack([Tn, Tn, Tn]).unsqueeze(0)
    print(Ts.shape)
    return Ts


class ClickNet(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.d = dimensions
        # self.conv = resnet18().to(device)
        self.conv = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        for param in self.conv.parameters():
            param.requires_grad = False

        self.conv.fc = nn.Linear(self.conv.fc.in_features, 1)

        # self.Layers = nn.Sequential(nn.Linear(1000, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(
        # ), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def convs(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        #x = self.Layers(x)
        return x  # torch.sigmoid(x)


def Train(project_name: str, force_rebuild=False, batch_size=8, epochs=5, dimensions=(int(540 / 2), int(960 / 2))):

    train_set = OsuDataset(project_name=project_name, force_rebuild=False)

    osu_data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    net = ClickNet(dimensions=dimensions).to(device)
    optimzer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.BCELoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, sample in tqdm(enumerate(osu_data_loader, 0)):
            data, result = sample

            net.zero_grad()
            outputs = net(data.to(device))
            loss = loss_function(outputs.float(), result.to(device).float())
            loss.backward()
            optimzer.step()
        print("LOSS", loss)

    correct = 0.0
    total = len(train_set.data)
    print('Testing')
    with torch.no_grad():
        for i in range(len(train_set.data)):
            if int(train_set.results[i][0]) == int(net(
                    torch.stack([trans(train_set.data[i])]).to(device))[0][0]):
                correct += 1

    print('Got {} Correct Out Of {}, {}% Accuracy'.format(
        correct, total, (correct/total) * 100))


Train('test-1')
