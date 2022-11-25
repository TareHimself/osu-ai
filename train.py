import time
from os import getcwd, path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dataset import OsuDataset
import torchvision.transforms as transforms
from models import ClicksNet, MouseNet
from torch.utils.data import DataLoader
from functorch import vmap

transform = transforms.ToTensor()

PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_PATH = path.normpath(path.join(getcwd(), 'models'))


def train_loop(model, data_loader, learning_rate, criterion, project_name, dataset_name, total_epochs, label_type: torch.Type = torch.LongTensor, get_accuracy=lambda predicted, actual: (predicted.argmax(1) == actual).sum().item()):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)

    for epoch in range(total_epochs):
        loading_bar = tqdm(total=len(data_loader))
        total_accu, total_count = 0, 0
        for idx, data in enumerate(data_loader):
            images, results = data
            images = images.type(torch.FloatTensor).to(PYTORCH_DEVICE)
            results = results.type(label_type).to(PYTORCH_DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, results)

            loss.backward()
            optimizer.step()
            total_accu += get_accuracy(outputs, results)
            total_count += results.size(0)
            loading_bar.set_description_str(
                f'Training {project_name} | Dataset {dataset_name} | epoch {epoch + 1}/{total_epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {loss.item():.4f} | ')
            loading_bar.update()
        loading_bar.close()


def train_action_net(dataset: str, force_rebuild=False, checkpoint_model=None, save_path=SAVE_PATH, batch_size=4,
                     epochs=1, learning_rate=0.0001, project_name=""):
    if len(project_name.strip()) == 0:
        project_name = dataset

    train_set = OsuDataset(project_name=dataset,
                           frame_latency=3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print(np.transpose(train_set[0][0],(3,270,280)).shape)

    osu_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    model = ClicksNet().to(device)

    if checkpoint_model:
        try:
            data = torch.load(path.normpath(
                path.join(save_path, f"{checkpoint_model}.pt")))
            model.load_state_dict(data['model'])
        except:
            pass

    criterion = nn.CrossEntropyLoss()

    train_loop(model, osu_data_loader, learning_rate,
               criterion, project_name, dataset, epochs)

    data = {
        'state': model.state_dict()
    }

    torch.save(data, path.normpath(
        path.join(save_path, f"model_action_{project_name}.pt")))


def train_aim_net(dataset: str, force_rebuild=False, checkpoint_model=None, save_path=SAVE_PATH, batch_size=4,
                  epochs=1, learning_rate=0.0001, project_name=""):

    if len(project_name.strip()) == 0:
        project_name = dataset

    train_set = OsuDataset(project_name=dataset,
                           frame_latency=-4, train_actions=False)

    osu_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    model = MouseNet().to(PYTORCH_DEVICE)

    if checkpoint_model:
        try:
            data = torch.load(path.normpath(
                path.join(save_path, f"{checkpoint_model}.pt")))
            model.load_state_dict(data['model'])
        except:
            pass

    criterion = nn.MSELoss()

    # accuracy is calculated with a 5 pixel error
    train_loop(model, osu_data_loader, learning_rate,
               criterion, project_name, dataset, epochs, label_type=torch.FloatTensor, get_accuracy=lambda predicted, actual: ((predicted - actual)**2).mean())

    data = {
        'state': model.state_dict()
    }

    torch.save(data, path.normpath(
        path.join(save_path, f"model_aim_{project_name}.pt")))


train_aim_net('meaning-of-love-4.62', checkpoint_model=None,
              epochs=10)
