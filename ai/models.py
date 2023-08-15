import torch
import torch.nn as nn
import uuid
import os
import json
import timm
from typing import Callable
from datetime import datetime
from ai.constants import CURRENT_STACK_NUM, FINAL_PLAY_AREA_SIZE
from ai.utils import refresh_model_list
from ai.enums import EModelType


def get_timm_model(build_final_layer: Callable[[int], nn.Module] = lambda a: nn.Linear(a, 2), channels=3,
                   model_name="resnet18", pretrained=False):
    model = timm.create_model(model_name=model_name, pretrained=pretrained, in_chans=channels, num_classes=3)
    # model = timm.create_model("resnet18",pretrained=True,in_chans=3,num_classes=3)
    classifier = model.default_cfg['classifier']

    in_features = getattr(model, classifier).in_features  # Get the number of input features for the final layer

    setattr(model, classifier, build_final_layer(in_features))  # Replace the final layer

    return model


class OsuAiModel(torch.nn.Module):
    def __init__(self, channels=CURRENT_STACK_NUM, model_type: EModelType = EModelType.Unknown) -> None:
        super().__init__()
        self.channels = channels
        self.model_type = model_type

    def save(self, project_name: str, datasets: list[str], epochs: int, learning_rate: int, path: str = './models',
             weights=None):

        model_id = str(uuid.uuid4())

        weights_to_save = weights if weights is not None else self.state_dict()

        save_dir = os.path.join(path, model_id)

        os.mkdir(save_dir)

        weights_dir = os.path.join(save_dir, 'weights.pt')

        model_dir = os.path.join(save_dir, 'model.pt')

        torch.save(weights_to_save, weights_dir)

        model_scripted = torch.jit.script(self)  # Export to TorchScript

        model_scripted.save(model_dir)  # Save
        
        config = {
            "name": project_name,
            "channels": self.channels,
            "date": str(datetime.utcnow()),
            "datasets": datasets,
            "type": self.model_type.name,
            "epochs": epochs,
            "lr": learning_rate
        }

        with open(os.path.join(save_dir, 'info.json'), 'w') as f:
            json.dump(config, f, indent=2)

        refresh_model_list()

    @staticmethod
    def load(model_id: str, model_gen=lambda *a, **b: OsuAiModel(*a, **b)):
        weights_path = os.path.join('./models', model_id, 'weights.pt')
        config_path = os.path.join('./models', model_id, 'info.json')
        weights = torch.load(weights_path)
        with open(config_path, 'r') as f:
            config_json = json.load(f)
            model = model_gen(
                channels=config_json['channels'], model_type=EModelType(config_json['type']))
            model.load_state_dict(weights)
            print(model.model_type)
            return model


class AimNet(OsuAiModel):
    """
    Works

    Args:
        torch (_type_): _description_
    """

    def __init__(self, channels=CURRENT_STACK_NUM, model_type: EModelType = EModelType.Aim):
        super().__init__(channels, model_type)

        self.conv = get_timm_model(build_final_layer=lambda features: nn.Sequential(
            nn.Linear(features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        ), channels=channels)

    def forward(self, images):
        return self.conv(images)

    @staticmethod
    def load(model_id: str):
        return OsuAiModel.load(model_id, lambda *a, **b: AimNet(*a, **b))


class ActionsNet(OsuAiModel):
    """
    Works so far

    Args:
        torch (_type_): _description_
    """

    def __init__(self, channels=CURRENT_STACK_NUM, model_type: EModelType = EModelType.Actions):
        super().__init__(channels, model_type)

        self.conv = get_timm_model(build_final_layer=lambda features: nn.Sequential(
            nn.Linear(features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 3),
        ), channels=channels)

    def forward(self, images):
        return self.conv(images)

    @staticmethod
    def load(model_id: str):
        return OsuAiModel.load(model_id, lambda *a, **b: ActionsNet(*a, **b))


class CombinedNet(OsuAiModel):
    """
    Works

    Args:
        torch (_type_): _description_
    """

    def __init__(self, channels=CURRENT_STACK_NUM, model_type: EModelType = EModelType.Combined):
        super().__init__(channels, model_type)

        self.conv = get_timm_model(build_final_layer=lambda features: nn.Sequential(
            nn.Linear(features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 4)
        ), channels=channels)

    def forward(self, images):
        return self.conv(images)

    @staticmethod
    def load(model_id: str):
        return OsuAiModel.load(model_id, lambda *a, **b: CombinedNet(*a, **b))