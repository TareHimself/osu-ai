import time
from os import getcwd, path
import cv2
import numpy as np
import torch
from functorch import vmap
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dataset import OsuDataset
import torchvision.transforms as transforms
from models import ActionsNet, AimNet
from torch.utils.data import DataLoader


transform = transforms.ToTensor()

PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_PATH = path.normpath(path.join(getcwd(), 'models'))


def train_action_net(dataset: str, force_rebuild=False, checkpoint_model=None, save_path=SAVE_PATH, batch_size=4, epochs=1, learning_rate=0.0001, project_name=""):

    if len(project_name.strip()) == 0:
        project_name = dataset

    train_set = OsuDataset(project_name=dataset, frame_latency=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    osu_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    # print(train_set[0][0].shape, train_set[1000:1050][1])

    model = ActionsNet().to(device)

    if checkpoint_model:
        try:
            data = torch.load(path.normpath(
                path.join(save_path, f"{checkpoint_model}.pt")))
            model.load_state_dict(data['model'])
        except:
            pass

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loading_bar = tqdm(total=len(osu_data_loader))
        total_accu, total_count = 0, 0
        running_loss = 0
        for idx, data in enumerate(osu_data_loader):
            images, results = data
            images = images.to(device)
            results = results.type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, results)

            loss.backward()
            optimizer.step()
            total_accu += (outputs.argmax(1) == results).sum().item()
            total_count += results.size(0)
            running_loss += loss.item() * images.size(0)
            loading_bar.set_description_str(
                f'Training Actions {project_name} | Dataset {dataset} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {loss.item():.4f} | ')
            loading_bar.update()
        loading_bar.set_description_str(
            f'Training Actions {project_name} | Dataset {dataset} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(osu_data_loader.dataset)):.4f} | ')
        loading_bar.close()

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
                           frame_latency=0, train_actions=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    osu_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    # print(train_set[0][0].shape, train_set[1000:1050][1])

    model = AimNet().to(device)

    if checkpoint_model:
        try:
            data = torch.load(path.normpath(
                path.join(save_path, f"{checkpoint_model}.pt")))
            model.load_state_dict(data['model'])
        except:
            pass

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loading_bar = tqdm(total=len(osu_data_loader))
        # total_accu, total_count = 0, 0
        running_loss = 0
        for idx, data in enumerate(osu_data_loader):
            images, results = data
            images = images.to(device)
            results = results.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, results)

            loss.backward()
            optimizer.step()
            # total_accu += (outputs.argmax(1) == results).sum().item()
            # total_count += results.size(0)
            running_loss += loss.item() * images.size(0)
            loading_bar.set_description_str(
                f'Training Aim {project_name} | Dataset {dataset} | epoch {epoch + 1}/{epochs} | loss {loss.item():.10f} | ')
            loading_bar.update()
        loading_bar.set_description_str(
            f'Training Aim {project_name} | Dataset {dataset} | epoch {epoch + 1}/{epochs} | loss {(running_loss / len(osu_data_loader.dataset)):.10f} | ')
        loading_bar.close()

    data = {
        'state': model.state_dict()
    }

    torch.save(data, path.normpath(
        path.join(save_path, f"model_aim_{project_name}.pt")))


train_aim_net('leia-2-5.41', checkpoint_model=None,
              epochs=20)

# train_action_net('leia-2-5.41', checkpoint_model=None,
#                  epochs=20)
