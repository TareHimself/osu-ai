import copy
from os import getcwd, path
import torch
import torch.nn as nn
from tqdm import tqdm
from ai.dataset import OsuDataset
import torchvision.transforms as transforms
from ai.models import ActionsNet, AimNet, TestModel
from torch.utils.data import DataLoader
from utils import get_datasets, get_validated_input, get_models
from constants import PYTORCH_DEVICE
import time

transform = transforms.ToTensor()

PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_PATH = path.normpath(path.join(getcwd(), 'models'))


def compute_element_wise_accuracy(pred, target, radius: int):
    total = 0
    total_correct = 0
    for x in range(len(pred)):
        coord_pred = pred[x]
        coord_target = target[x]
        if coord_target[0] - radius <= coord_pred[0] <= coord_target[0] + radius and coord_target[1] - radius <= coord_pred[1] <= coord_target[1] + radius:
            total_correct += 1

        total += 1

    return total, total_correct


def train_action_net(datasets: list[str], force_rebuild=False, checkpoint_model_id=None, save_path=SAVE_PATH, batch_size=32,
                     epochs=1, learning_rate=0.0003, project_name=""):

    if len(project_name.strip()) == 0:
        project_name = "-".join(map(lambda a: a[:-4], datasets))

    train_set = OsuDataset(datasets=datasets, frame_latency=0)

    osu_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    model = None

    if checkpoint_model_id:
        model = ActionsNet.load(checkpoint_model_id).type(
            torch.FloatTensor).to(PYTORCH_DEVICE)
    else:
        model = ActionsNet().type(torch.FloatTensor).to(PYTORCH_DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = 999999999999
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

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
                    f'Training Actions Using {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(osu_data_loader.dataset)):.4f} | ')
                loading_bar.update()
            loading_bar.set_description_str(
                f'Training Actions Using {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(osu_data_loader.dataset)):.4f} | ')
            loading_bar.close()

            if running_loss / len(osu_data_loader.dataset) < best_loss:
                best_loss = running_loss / len(osu_data_loader.dataset)
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

    except KeyboardInterrupt:
        if get_validated_input("Would you like to save the last epoch?\n", lambda a: True, lambda a: a.strip().lower()).startswith("y"):
            model.save(project_name, best_epoch,
                       learning_rate, weights=best_state)
        return

    model.save(project_name, best_epoch, learning_rate, weights=best_state)


def train_aim_net(datasets: str, force_rebuild=False, checkpoint_model_id=None, save_path=SAVE_PATH, batch_size=64,
                  epochs=1, learning_rate=0.0001, project_name=""):
    if len(project_name.strip()) == 0:
        project_name = "-".join(map(lambda a: a[:-4], datasets))

    train_set = OsuDataset(datasets=datasets,
                           frame_latency=0, label_type=2)

    osu_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    # print(train_set[0][0].shape, train_set[1000:1050][1])

    model = None

    if checkpoint_model_id:
        model = AimNet.load(checkpoint_model_id).to(PYTORCH_DEVICE)
    else:
        model = AimNet().to(PYTORCH_DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    MAX_ERROR = 0.001

    CLAMP_MIN = torch.Tensor([0]).to(PYTORCH_DEVICE)

    CLAMP_MAX = torch.Tensor([1]).to(PYTORCH_DEVICE)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = 99999999999
    best_epoch = 0
    patience = 50
    patience_count = 0
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

                total_calc_size, total_correct_calc = compute_element_wise_accuracy(
                    outputs, expected, 0.01)
                total_accu += total_correct_calc
                total_count += total_calc_size
                running_loss += loss.item() * images.size(0)
                loading_bar.set_description_str(
                    f'Training Aim Using {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(osu_data_loader.dataset)):.10f} | ')
                loading_bar.update()
            epoch_loss = running_loss / len(osu_data_loader.dataset)
            epoch_accu = (total_accu / total_count) * 100
            loading_bar.set_description_str(
                f'Training Aim Using {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {(epoch_accu):.4f} | loss {(epoch_loss):.10f} | ')
            loading_bar.close()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == patience:
                model.save(project_name, best_epoch,
                           learning_rate, weights=best_state)
                return
    except KeyboardInterrupt:
        if get_validated_input("Would you like to save the best epoch?\n", lambda a: True, lambda a: a.strip().lower()).startswith("y"):
            model.save(project_name, best_epoch,
                       learning_rate, weights=best_state)

        return

    model.save(project_name, best_epoch, learning_rate, weights=best_state)


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
    print('\nConfig for', data_type, '\n')
    epochs = get_validated_input("Max epochs to train for ?\n",
                                 lambda a: a.strip().isnumeric() and 0 <= int(a.strip()), lambda a: int(a.strip()))

    if get_validated_input("Would you like to use a checkpoint?\n", lambda a: True, lambda a: a.strip().lower()).startswith("y"):
        checkpoint_index = get_validated_input(models_prompt, lambda a: a.strip().isnumeric() and (
            0 <= int(a.strip()) < len(models_prompt)), lambda a: int(a.strip()))
        checkpoint = models[checkpoint_index]

    return data_type, list(map(lambda a: datasets[a], selected_datasets)), checkpoint, epochs


def start_train():
    datasets = get_datasets()
    prompt = """What type of training would you like to do?
    [0] Train Aim 
    [1] Train Actions
    [2] Train Both
"""

    dataset_prompt = "Please select datasets from below seperated by a comma:\n"

    for i in range(len(datasets)):
        dataset_prompt += f"    [{i}] {datasets[i]}\n"

    user_choice = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
        0 <= int(a.strip()) <= 2), lambda a: int(a.strip()))

    training_tasks = []

    if user_choice == 0 or user_choice == 2:
        models_prompt = "Please select a model from below:\n"

        models = get_models('aim')

        for i in range(len(models)):
            models_prompt += f"    [{i}] {models[i]}\n"

        training_tasks.append(get_train_data(
            'aim', datasets, dataset_prompt, models, models_prompt))

    if user_choice == 1 or user_choice == 2:
        models_prompt = "Please select a model from below:\n"

        models = get_models('action')

        for i in range(len(models)):
            models_prompt += f"    [{i}] {models[i]}\n"

        training_tasks.append(get_train_data(
            'actions', datasets, dataset_prompt, models, models_prompt))

    for task in training_tasks:
        task_type, dataset, checkpoint, epochs = task
        if task_type == 'aim':
            train_aim_net(
                dataset, False, checkpoint[0] if checkpoint is not None else None, epochs=epochs)
        else:
            train_action_net(
                dataset, False, checkpoint[0] if checkpoint is not None else None, epochs=epochs)
