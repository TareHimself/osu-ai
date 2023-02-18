from os import getcwd, path
import torch
import torch.nn as nn
from tqdm import tqdm
from ai.dataset import OsuDataset
import torchvision.transforms as transforms
from ai.models import ActionsNet, AimNet
from torch.utils.data import DataLoader
from utils import get_datasets, get_model_path, get_validated_input, get_models
from constants import PYTORCH_DEVICE
import time

transform = transforms.ToTensor()

PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_PATH = path.normpath(path.join(getcwd(), 'models'))


def train_action_net(datasets: list[str], force_rebuild=False, checkpoint_model=None, save_path=SAVE_PATH, batch_size=4,
                     epochs=1, learning_rate=0.0001, project_name=""):

    if len(project_name.strip()) == 0:
        project_name = "-".join(datasets)

    train_set = OsuDataset(datasets=datasets, frame_latency=2)

    osu_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    model = ActionsNet().type(torch.FloatTensor).to(PYTORCH_DEVICE)

    if checkpoint_model:
        model.load(get_model_path(checkpoint_model))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    last_state_dict = model.state_dict()

    try:
        for epoch in range(epochs):
            loading_bar = tqdm(total=len(osu_data_loader))
            total_accu, total_count = 0, 0
            running_loss = 0
            for idx, data in enumerate(osu_data_loader):
                images, results = data
                images = images.type(torch.FloatTensor).to(PYTORCH_DEVICE)
                results = results.type(torch.LongTensor).to(PYTORCH_DEVICE)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, results)

                loss.backward()
                optimizer.step()
                total_accu += (outputs.argmax(1) == results).sum().item()
                total_count += results.size(0)
                running_loss += loss.item() * images.size(0)
                loading_bar.set_description_str(
                    f'Training Actions Using {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {loss.item():.4f} | ')
                loading_bar.update()
            loading_bar.set_description_str(
                f'Training Actions Using {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(osu_data_loader.dataset)):.4f} | ')
            loading_bar.close()
            last_state_dict = model.state_dict()

    except KeyboardInterrupt:
        if get_validated_input("Would you like to save the last epoch?\n", lambda a: True, lambda a: a.strip().lower()).startswith("y"):
            model.save(path.normpath(
                path.join(save_path, f"model_aim_{project_name}_{time.strftime('%d-%m-%y-%H-%M-%S')}.pt")), {
                'state': last_state_dict
            })
        return

    model.save(path.normpath(
        path.join(save_path, f"model_aim_{project_name}_{time.strftime('%d-%m-%y-%H-%M-%S')}.pt")))


def train_aim_net(datasets: str, force_rebuild=False, checkpoint_model=None, save_path=SAVE_PATH, batch_size=32,
                  epochs=1, learning_rate=0.0003, project_name=""):
    if len(project_name.strip()) == 0:
        project_name = "-".join(datasets)

    train_set = OsuDataset(datasets=datasets,
                           frame_latency=0, is_actions=False)

    osu_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    # print(train_set[0][0].shape, train_set[1000:1050][1])

    model = AimNet().to(PYTORCH_DEVICE)

    if checkpoint_model:
        model.load(get_model_path(checkpoint_model))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    MAX_ERROR = 0.001

    CLAMP_MIN = torch.Tensor([0]).to(PYTORCH_DEVICE)

    CLAMP_MAX = torch.Tensor([1]).to(PYTORCH_DEVICE)

    last_state_dict = model.state_dict()
    try:
        for epoch in range(epochs):
            loading_bar = tqdm(total=len(osu_data_loader))
            total_accu, total_count = 0, 0
            running_loss = 0
            for idx, data in enumerate(osu_data_loader):
                images, expected = data
                images: torch.Tensor = images.type(
                    torch.FloatTensor).to(PYTORCH_DEVICE)
                expected: torch.Tensor = expected.type(
                    torch.FloatTensor).to(PYTORCH_DEVICE)

                optimizer.zero_grad()

                outputs: torch.Tensor = model(images)

                loss = criterion(outputs, expected)

                loss.backward()
                optimizer.step()

                total_accu += (((outputs - expected).absolute() <
                                MAX_ERROR).sum(dim=1) - 1).clamp(CLAMP_MIN, CLAMP_MAX).sum().item()
                total_count += expected.size(0)
                running_loss += loss.item() * images.size(0)
                loading_bar.set_description_str(
                    f'Training Aim Using {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(loss.item()):.10f} | ')
                loading_bar.update()
            loading_bar.set_description_str(
                f'Training Aim Using {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(osu_data_loader.dataset)):.10f} | ')
            loading_bar.close()
            last_state_dict = model.state_dict()
    except KeyboardInterrupt:
        if get_validated_input("Would you like to save the last epoch?\n", lambda a: True, lambda a: a.strip().lower()).startswith("y"):
            model.save(path.normpath(
                path.join(save_path, f"model_aim_{project_name}_{time.strftime('%d-%m-%y-%H-%M-%S')}.pt")), {
                'state': last_state_dict
            })

        return

    model.save(path.normpath(
        path.join(save_path, f"model_aim_{project_name}_{time.strftime('%d-%m-%y-%H-%M-%S')}.pt")))


def get_train_data(data_type, datasets, datasets_prompt, models, models_prompt):
    def validate_datasets_selection(reciev: str):
        try:
            items = map(int, reciev.strip().split(","))
            for item in items:
                if not 0 <= item < len(datasets):
                    return False
            return True
        except:
            return False

    selected_datasets = get_validated_input(
        datasets_prompt, validate_datasets_selection, lambda a: map(int, a.strip().split(",")))
    checkpoint = None
    epochs = get_validated_input("How many epochs would you like to train for ?\n",
                                 lambda a: a.strip().isnumeric() and 0 <= int(a.strip()), lambda a: int(a.strip()))

    if get_validated_input("Would you like to use a checkpoint?\n", lambda a: True, lambda a: a.strip().lower()).startswith("y"):
        checkpoint_index = get_validated_input(models_prompt, lambda a: a.strip().isnumeric() and (
            0 <= int(a.strip()) < len(models_prompt)), lambda a: int(a.strip()))
        checkpoint = models[checkpoint_index]

    return data_type, list(map(lambda a: datasets[a], selected_datasets)), checkpoint, epochs


def start_train():
    datasets = get_datasets()
    models = get_models()
    prompt = """What type of training would you like to do?
    [0] Train Aim 
    [1] Train Actions
    [2] Train Both
"""

    dataset_prompt = "Please select datasets from below seperated by a comma:\n"
    models_prompt = "Please select a model from below:\n"
    for i in range(len(datasets)):
        dataset_prompt += f"    [{i}] {datasets[i]}\n"

    for i in range(len(models)):
        models_prompt += f"    [{i}] {models[i]}\n"

    user_choice = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
        0 <= int(a.strip()) <= 2), lambda a: int(a.strip()))

    training_tasks = []

    if user_choice == 0 or user_choice == 2:
        training_tasks.append(get_train_data(
            'aim', datasets, dataset_prompt, models, models_prompt))

    if user_choice == 1 or user_choice == 2:
        training_tasks.append(get_train_data(
            'actions', datasets, dataset_prompt, models, models_prompt))

    for task in training_tasks:
        task_type, dataset, checkpoint, epochs = task
        if task_type == 'aim':
            train_aim_net(dataset, False, checkpoint, epochs=epochs)
        else:
            train_action_net(dataset, False, checkpoint, epochs=epochs)
