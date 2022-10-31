import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dataset import OsuDataset
import torchvision.transforms as transforms
from torchvision.models import resnet18
transform = transforms.ToTensor()

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def numpyToTensor(arr, d):
    print(arr.shape)
    return transform(arr)
    print(arr.shape)
    Tn = torch.Tensor(d[0], d[1])
    Ts = torch.stack([Tn, Tn, Tn]).unsqueeze(0)
    print(Ts.shape)
    return Ts


class Net(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.d = dimensions
        # self.conv = resnet18().to(device)
        self.conv = resnet18(pretrained=True).to(device)
        for param in self.conv.parameters():
            param.requires_grad = False

        self._lin = None
        self.fc1 = None
        self.fc2 = nn.Linear(1024, 4)

    def _calc_linear_size(self, val):
        ret = val.shape
        return ret[1]

    def convs(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def forward(self, x: torch.Tensor):

        x = self.convs(x)
        if self._lin is None:
            self._lin = self._calc_linear_size(x)
            self.fc1 = nn.Linear(self._lin, 1024).to(device)
        x = x.view(-1, self._lin)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def Train(project_name: str, force_rebuild=False, batch_size=4, epochs=15, dimensions=(int(540 / 2), int(960 / 2))):

    train_set = OsuDataset(project_name=project_name)

    osu_data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    net = Net(dimensions=dimensions).to(device)

    optimzer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

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

    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(10):
            x = i + 30
            print("ACTUAL:", train_set.results[x], 'PREDICTED:', net(
                torch.stack([transform(train_set.data[x])]).to(device)))


Train('osu-rgb')
