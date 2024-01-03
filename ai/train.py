import copy
from os import getcwd, path
import torch
import torch.nn as nn
from tqdm import tqdm
from ai.dataset import OsuDataset
import torchvision.transforms as transforms
from ai.models import ActionsNet, AimNet, CombinedNet
from torch.utils.data import DataLoader
from ai.utils import get_datasets, get_validated_input, get_models
from ai.constants import PYTORCH_DEVICE, SCREEN_HEIGHT, SCREEN_WIDTH
from ai.enums import EModelType

transform = transforms.ToTensor()

PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_PATH = path.normpath(path.join(getcwd(), "models"))


@torch.jit.script
def get_acc(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    screen_width: int = SCREEN_WIDTH,
    screen_height: int = SCREEN_HEIGHT,
    thresh: int = 60,
    is_combined: bool = False,
):
    predicted = predicted.detach().clone()
    truth = truth.detach().clone()

    predicted[:, 0] *= screen_width
    predicted[:, 1] *= screen_height
    truth[:, 0] *= screen_width
    truth[:, 1] *= screen_height

    diff = (predicted[:, :-2] - truth[:, :-2]) if is_combined else predicted - truth

    dist = torch.sqrt((diff**2).sum(dim=1))

    dist[dist < thresh] = 1

    dist[dist >= thresh] = 0

    if not is_combined:
        return dist.mean().item()

    predicted_keys = predicted[:, 2:]
    truth_keys = truth[:, 2:]

    predicted_keys[predicted_keys >= 0.5] = 1
    truth_keys[truth_keys >= 0.5] = 1
    predicted_keys[predicted_keys < 0.5] = 0
    truth_keys[truth_keys < 0.5] = 0

    return (
        dist.mean().item()
        + torch.all(predicted_keys == truth_keys, dim=1).float().mean().item()
    ) / 2


def train_action_net(
    datasets: list[str],
    force_rebuild=False,
    checkpoint_model_id=None,
    batch_size=64,
    epochs=1,
    learning_rate=0.0001,
    project_name="",
):
    if len(project_name.strip()) == 0:
        project_name = f"Project with {len(datasets)} Datasets"

    train_set = OsuDataset(datasets=datasets, label_type=EModelType.Actions)

    osu_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = None

    if checkpoint_model_id:
        model = (
            ActionsNet.load(checkpoint_model_id)
            .type(torch.FloatTensor)
            .to(PYTORCH_DEVICE)
        )
    else:
        model = ActionsNet().type(torch.FloatTensor).to(PYTORCH_DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = 99999999999
    best_epoch = 0
    patience = 20
    patience_count = 0

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
                    f"Training Actions | {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(osu_data_loader.dataset)):.8f} | "
                )
                loading_bar.update()
            loading_bar.set_description_str(
                f"Training Actions | {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(osu_data_loader.dataset)):.8f} | "
            )
            loading_bar.close()
            epoch_loss = running_loss / len(osu_data_loader.dataset)
            epoch_accu = (total_accu / total_count) * 100

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == patience:
                break

            # if running_loss / len(osu_data_loader.dataset) < best_loss:
            #     best_loss = running_loss / len(osu_data_loader.dataset)
            #     best_state = copy.deepcopy(model.state_dict())
            #     best_epoch = epoch

    except KeyboardInterrupt:
        if not get_validated_input(
            "Would you like to save the best epoch?\n",
            lambda a: True,
            lambda a: a.strip().lower(),
        ).startswith("y"):
            return

    model.load_state_dict(best_state)
    model.save(project_name, datasets, best_epoch, learning_rate)


def train_aim_net(
    datasets: list[str],
    force_rebuild=False,
    checkpoint_model_id=None,
    batch_size=64,
    epochs=1,
    learning_rate=0.0001,
    project_name="",
):
    if len(project_name.strip()) == 0:
        project_name = f"Project with {len(datasets)} Datasets"

    train_set = OsuDataset(datasets=datasets, label_type=EModelType.Aim)

    osu_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # print(train_set[0][0].shape, train_set[1000:1050][1])

    model = None

    if checkpoint_model_id:
        model = AimNet.load(checkpoint_model_id).to(PYTORCH_DEVICE)
    else:
        model = AimNet().to(PYTORCH_DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = 99999999999
    best_epoch = 0
    patience = 100
    patience_count = 0
    try:
        for epoch in range(epochs):
            loading_bar = tqdm(total=len(osu_data_loader))
            total_accu, total_count = 0, 0
            running_loss = 0
            for idx, data in enumerate(osu_data_loader):
                images, expected = data
                images: torch.Tensor = images.type(torch.FloatTensor).to(PYTORCH_DEVICE)
                expected: torch.Tensor = expected.type(torch.FloatTensor).to(
                    PYTORCH_DEVICE
                )

                outputs: torch.Tensor = model(images)

                loss = criterion(outputs, expected)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                total_accu += get_acc(outputs, expected)
                total_count += 1
                running_loss += loss.item() * images.size(0)
                loading_bar.set_description_str(
                    f"Training Aim | {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(osu_data_loader.dataset)):.10f} | "
                )
                loading_bar.update()
            epoch_loss = running_loss / len(osu_data_loader.dataset)
            epoch_accu = (total_accu / total_count) * 100
            loading_bar.set_description_str(
                f"Training Aim | {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {(epoch_accu):.4f} | loss {(epoch_loss):.10f} | "
            )
            loading_bar.close()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == patience:
                break

    except KeyboardInterrupt:
        if not get_validated_input(
            "Would you like to save the best epoch?\n",
            lambda a: True,
            lambda a: a.strip().lower(),
        ).startswith("y"):
            return

    model.load_state_dict(best_state)
    model.save(project_name, datasets, best_epoch, learning_rate)


def train_combined_net(
    datasets: list[str],
    force_rebuild=False,
    checkpoint_model_id=None,
    batch_size=64,
    epochs=1,
    learning_rate=0.0001,
    project_name="",
):
    if len(project_name.strip()) == 0:
        project_name = f"Project with {len(datasets)} Datasets"

    train_set = OsuDataset(datasets=datasets, label_type=EModelType.Combined)

    osu_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # print(train_set[0][0].shape, train_set[1000:1050][1])

    model = None

    if checkpoint_model_id:
        model = CombinedNet.load(checkpoint_model_id).to(PYTORCH_DEVICE)
    else:
        model = CombinedNet().to(PYTORCH_DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = 99999999999
    best_epoch = 0
    patience = 100
    patience_count = 0
    try:
        for epoch in range(epochs):
            loading_bar = tqdm(total=len(osu_data_loader))
            total_accu, total_count = 0, 0
            running_loss = 0
            for idx, data in enumerate(osu_data_loader):
                images, expected = data
                images: torch.Tensor = images.type(torch.FloatTensor).to(PYTORCH_DEVICE)
                expected: torch.Tensor = expected.type(torch.FloatTensor).to(
                    PYTORCH_DEVICE
                )

                outputs: torch.Tensor = model(images)

                loss = criterion(outputs, expected)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                total_accu += get_acc(outputs, expected, is_combined=True)
                total_count += 1
                running_loss += loss.item() * images.size(0)
                loading_bar.set_description_str(
                    f"Training Aim | {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(osu_data_loader.dataset)):.10f} | "
                )
                loading_bar.update()
            epoch_loss = running_loss / len(osu_data_loader.dataset)
            epoch_accu = (total_accu / total_count) * 100
            loading_bar.set_description_str(
                f"Training Aim | {project_name} | epoch {epoch + 1}/{epochs} |  Accuracy {(epoch_accu):.4f} | loss {(epoch_loss):.10f} | "
            )
            loading_bar.close()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == patience:
                break

    except KeyboardInterrupt:
        if not get_validated_input(
            "Would you like to save the best epoch?\n",
            lambda a: True,
            lambda a: a.strip().lower(),
        ).startswith("y"):
            return

    model.load_state_dict(best_state)
    model.save(project_name, datasets, best_epoch, learning_rate)


def get_train_data(
    data_type: EModelType,
    datasets: list[int],
    datasets_prompt: str,
    models: list[dict],
    models_prompt: str,
):
    project_name = get_validated_input(
        "What would you like to name this project ?",
        lambda a: True,
        lambda a: a.strip(),
    )

    def validate_datasets_selection(received: str):
        try:
            items = map(int, received.strip().split(","))
            for item in items:
                if not 0 <= item < len(datasets):
                    return False
            return True
        except:
            return False

    selected_datasets = get_validated_input(
        datasets_prompt,
        validate_datasets_selection,
        lambda a: map(int, a.strip().split(",")),
    )
    checkpoint = None
    print("\nConfig for", data_type.value, "\n")
    epochs = get_validated_input(
        "Max epochs to train for ?\n",
        lambda a: a.strip().isnumeric() and 0 <= int(a.strip()),
        lambda a: int(a.strip()),
    )

    if len(models) > 0:
        if get_validated_input(
            "Would you like to use a checkpoint?\n",
            lambda a: True,
            lambda a: a.strip().lower(),
        ).startswith("y"):
            checkpoint_index = get_validated_input(
                models_prompt,
                lambda a: a.strip().isnumeric()
                and (0 <= int(a.strip()) < len(models_prompt)),
                lambda a: int(a.strip()),
            )
            checkpoint = models[checkpoint_index]

    return (
        data_type,
        project_name,
        list(map(lambda a: datasets[a], selected_datasets)),
        checkpoint,
        epochs,
    )


def start_train():
    datasets = get_datasets()
    prompt = """What type of training would you like to do?
    [0] Train Aim 
    [1] Train Actions
    [2] Train Combined
"""

    dataset_prompt = "Please select datasets from below seperated by a comma:\n"

    for i in range(len(datasets)):
        dataset_prompt += f"    [{i}] {datasets[i]}\n"

    user_choice = get_validated_input(
        prompt,
        lambda a: a.strip().isnumeric() and (0 <= int(a.strip()) <= 2),
        lambda a: int(a.strip()),
    )

    training_tasks = []

    models_prompt = "Please select a model from below:\n"

    if user_choice == 0:
        models = get_models(EModelType.Aim)

        for i in range(len(models)):
            models_prompt += f"    [{i}] {models[i]}\n"

        training_tasks.append(
            get_train_data(
                EModelType.Aim, datasets, dataset_prompt, models, models_prompt
            )
        )

    elif user_choice == 1:
        models = get_models(EModelType.Actions)

        for i in range(len(models)):
            models_prompt += f"    [{i}] {models[i]}\n"

        training_tasks.append(
            get_train_data(
                EModelType.Actions, datasets, dataset_prompt, models, models_prompt
            )
        )

    else:
        models = get_models(EModelType.Combined)

        for i in range(len(models)):
            models_prompt += f"    [{i}] {models[i]}\n"

        training_tasks.append(
            get_train_data(
                EModelType.Combined, datasets, dataset_prompt, models, models_prompt
            )
        )

    for task in training_tasks:
        task_type, project_name, dataset, checkpoint, epochs = task
        if task_type == EModelType.Aim:
            train_aim_net(
                project_name=project_name,
                datasets=dataset,
                checkpoint_model_id=checkpoint["id"]
                if checkpoint is not None
                else None,
                epochs=epochs,
            )
        elif task_type == EModelType.Actions:
            train_action_net(
                project_name=project_name,
                datasets=dataset,
                checkpoint_model_id=checkpoint["id"]
                if checkpoint is not None
                else None,
                epochs=epochs,
            )
        else:
            train_combined_net(
                project_name=project_name,
                datasets=dataset,
                checkpoint_model_id=checkpoint["id"]
                if checkpoint is not None
                else None,
                epochs=epochs,
            )
