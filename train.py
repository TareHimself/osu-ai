from os import getcwd, path
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import OsuDataset
import torchvision.transforms as transforms
from models import ActionsNet, AimNet
from torch.utils.data import DataLoader
from utils import get_datasets, get_validated_input, get_models
from constants import PYTORCH_DEVICE
from utils import load_model_data

transform = transforms.ToTensor()

PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_PATH = path.normpath(path.join(getcwd(), 'models'))


def train_action_net(dataset: str, force_rebuild=False, checkpoint_model=None, save_path=SAVE_PATH, batch_size=4,
                     epochs=1, learning_rate=0.0001, project_name=""):
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

    model = ActionsNet().to(PYTORCH_DEVICE)

    if checkpoint_model:
        data = load_model_data(checkpoint_model)
        model.load_state_dict(data['model'])

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

    osu_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    # print(train_set[0][0].shape, train_set[1000:1050][1])

    model = AimNet().to(PYTORCH_DEVICE)

    if checkpoint_model:
        data = load_model_data(checkpoint_model)
        model.load_state_dict(data['model'])

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

def get_train_data(datasets, datasets_prompt, models, models_prompt):
    selected_dataset = get_validated_input(datasets_prompt, lambda a: a.strip().isnumeric() and (
            0 <= int(a.strip()) < len(datasets)), lambda a: int(a.strip()))
    checkpoint = None
    epochs = get_validated_input("How many epochs would you like to train for ?",
                                 lambda a: a.strip().isnumeric() and 0 <= int(a.strip()), lambda a: int(a.strip()))

    if get_validated_input("Would you like to use a checkpoint?\n").lower().startswith("y"):
        checkpoint_index = get_validated_input(models_prompt, lambda a: a.strip().isnumeric() and (
                0 <= int(a.strip()) < len(models_prompt)), lambda a: int(a.strip()))
        checkpoint = models[checkpoint_index]


def train_new():
    datasets = get_datasets()
    models = get_models()
    prompt = """
What type of training would you like to do?
    [0] Train Aim 
    [1] Train Actions
    [2] Train Both
"""

    dataset_prompt = "Please select a dataset from below:\n"
    models_prompt = "Please select a model from below:\n"

    for i in range(len(datasets)):
        dataset_prompt += f"    [{i}] {datasets[i]}"

    for i in range(len(models)):
        models_prompt += f"    [{i}] {models[i]}"

    user_choice = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
            0 <= int(a.strip()) <= 2), lambda a: int(a.strip()))

    training_tasks = []

    if user_choice == 0 or user_choice == 2:
        training_tasks.append(get_train_data('aim', datasets, dataset_prompt, models, models_prompt))

    if user_choice == 1 or user_choice == 2:
        training_tasks.append(get_train_data('actions', datasets, dataset_prompt, models, models_prompt))

    for task in training_tasks:
        task_type, dataset, checkpoint, epochs = task
        if task_type == 'aim':
            train_aim_net(dataset, False, checkpoint, epochs=epochs)
        else:
            train_action_net(dataset, False, checkpoint, epochs=epochs)
