import time
from os import path
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
transform = transforms.ToTensor()

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def Train(project_name: str, force_rebuild=False, checkpoint_path=None, save_path="model.ptf", batch_size=4, epochs=1, learning_rate=0.0001):

    train_set = OsuDataset(project_name=project_name)  # , force_rebuild=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # image = train_set[0][0]
    # print(image.shape,image)

    # print(np.transpose(train_set[0][0],(3,270,280)).shape)

    osu_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    print(train_set[0][0].shape, train_set[1000:1050][1])

    model = ClicksNet().to(device)

    if checkpoint_path:
        try:
            data = torch.load(checkpoint_path)
            model.load_state_dict(data['model'])
        except:
            pass

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loading_bar = tqdm(total=len(osu_data_loader))
        total_accu, total_count = 0, 0
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
            loading_bar.set_description_str(
                f'Training {project_name} :: epoch {epoch + 1}/{epochs} ::  Accuracy {((total_accu / total_count) * 100):.4f} :: loss {loss.item():.4f} :')
            loading_bar.update()
        loading_bar.close()

    data = {
        'state': model.state_dict()
    }

    torch.save(data, save_path)


def test(model_path=None, image=""):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ClicksNet()
    model.to(device)
    model.load_state_dict(torch.load(model_path)['state'])
    model.eval()

    cv_image = cv2.imread(image, cv2.IMREAD_COLOR)

    start = time.time()
    image = transform(np.array(cv_image)).reshape(
        (1, 3, 270, 480)).to(device)

    output = model(image)

    _, predicated = torch.max(output, dim=1)

    end = time.time()

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicated.item()]

    return prob.item(), predicated.item(), end - start


Train('body-floating-5.77', checkpoint_path=None,
      save_path="body-floating-5.77.pt", epochs=20)
