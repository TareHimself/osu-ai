import torch
import torch.nn as nn
import uuid
import os
import json
import timm
import inspect
from typing import Callable
from datetime import datetime
from ai.constants import (
    CURRENT_STACK_NUM,
    FINAL_PLAY_AREA_SIZE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from ai.utils import refresh_model_list
from ai.enums import EModelType


def get_timm_model(
    build_final_layer: Callable[[int], nn.Module] = lambda a: nn.Linear(a, 2),
    channels=3,
    model_name="resnet18",
    pretrained=False,
):
    model = timm.create_model(
        model_name=model_name, pretrained=pretrained, in_chans=channels, num_classes=3
    )
    # model = timm.create_model("resnet18",pretrained=True,in_chans=3,num_classes=3)
    classifier = model.default_cfg["classifier"]

    in_features = getattr(
        model, classifier
    ).in_features  # Get the number of input features for the final layer

    setattr(
        model, classifier, build_final_layer(in_features)
    )  # Replace the final layer

    return model


class Module(torch.nn.Module):
    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        self.load_state_dict(torch.load(path))

    def get_source(self):
        return inspect.getsource(Module)


class FeatureExtractor(Module):
    def __init__(self, channels=3, out_features=1024) -> None:
        super().__init__()
        self.out_features = out_features
        self.channels = channels

    def forward(self, x):
        raise BaseException("Forward not implemented on extractor")

    def info_json(self) -> dict[str, str]:
        return {
            "channels": self.channels,
            "out_features": self.out_features,
            "source": self.get_source(),
        }

    def get_source(self):
        return inspect.getsource(FeatureExtractor)


class ResnetFeatureExtractor(FeatureExtractor):
    def __init__(self, channels=3, out_features=1024) -> None:
        super().__init__(channels=channels, out_features=out_features)
        self.resnet = get_timm_model(
            build_final_layer=lambda features: nn.Sequential(
                nn.Linear(features, out_features),
            ),
            channels=channels,
        )

    def forward(self, x):
        return self.resnet(x)

    def get_source(self):
        return inspect.getsource(ResnetFeatureExtractor)


class FC(Module):
    def __init__(self, channels=3, in_features=1024) -> None:
        super().__init__()
        self.in_features = in_features
        self.channels = channels

    def forward(self, x):
        raise BaseException("Forward not implemented on extractor")

    def get_source(self):
        return inspect.getsource(FC)

    def info_json(self) -> dict[str, str]:
        return {
            "channels": self.channels,
            "in_features": self.in_features,
            "source": inspect.getsource(self.__class__),
        }


class AimRegression(FC):
    def __init__(self, channels=3, in_features=1024) -> None:
        super().__init__(channels=channels, in_features=in_features)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.fc(x)

    def get_source(self):
        return inspect.getsource(AimRegression)


class ClickClassifier(FC):
    def __init__(self, channels=3, in_features=1024) -> None:
        super().__init__(channels=channels, in_features=in_features)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        return self.fc(x)

    def get_source(self):
        return inspect.getsource(ClickClassifier)


class CombinedOutput(FC):
    def __init__(self, channels=3, in_features=1024) -> None:
        super().__init__(channels=channels, in_features=in_features)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        return self.fc(x)

    def get_source(self):
        return inspect.getsource(CombinedOutput)


class OsuAiModel(Module):
    def __init__(
        self,
        channels=CURRENT_STACK_NUM,
        model_type: EModelType = EModelType.Unknown,
        num_features: int = 1024,
        extractor_factory: Callable[
            [int, int], FeatureExtractor
        ] = lambda channels, out_feat: FeatureExtractor(
            channels=channels, out_features=out_feat
        ),
        fc_factory: Callable[[int, int], FC] = lambda channels, out_feat: FC(
            channels=channels, out_features=out_feat
        ),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.model_type = model_type
        self.num_features = num_features
        self.extractor = extractor_factory(self.channels, self.num_features)
        self.fc = fc_factory(self.channels, self.extractor.out_features)

    def forward(self, images):
        return self.fc(self.extractor(images))

    def get_source(self):
        return inspect.getsource(OsuAiModel)

    def save(
        self,
        project_name: str,
        datasets: list[str],
        epochs: int,
        learning_rate: int,
        path: str = "./models",
    ):
        model_id = str(uuid.uuid4())

        # weights_to_save = self.state_dict()

        save_dir = os.path.join(path, model_id)

        os.mkdir(save_dir)

        config = {
            "name": project_name,
            "date": str(datetime.utcnow()),
            "datasets": datasets,
            "epochs": epochs,
            "type": self.model_type.name,
            "lr": learning_rate,
            "channels": self.channels,
            "screen_size": [SCREEN_WIDTH, SCREEN_HEIGHT],
            "input_size": FINAL_PLAY_AREA_SIZE,
            "fc": self.fc.info_json(),
            "extractor": self.extractor.info_json(),
            "source": self.get_source(),
        }

        with open(os.path.join(save_dir, "info.json"), "w") as f:
            json.dump(config, f, indent=2)

        extractor_dir = os.path.join(save_dir, "extractor.pt")
        fc_dir = os.path.join(save_dir, "fc.pt")
        all_dir = os.path.join(save_dir, "all.pt")
        self.extractor.save_weights(extractor_dir)
        self.fc.save_weights(fc_dir)
        self.save_weights(all_dir)

        model_scripted = torch.jit.script(self)  # Export to TorchScript
        model_dir = os.path.join(save_dir, "all.torchscript")
        model_scripted.save(model_dir)  # Save

        refresh_model_list()

    @staticmethod
    def load(
        model_id: str,
        model_factory: Callable[
            [int, EModelType], "OsuAiModel"
        ] = lambda *a, **b: OsuAiModel(*a, **b),
    ):
        model_folder = os.path.join("./models", model_id)

        config_path = os.path.join(model_folder, "info.json")

        with open(config_path, "r") as f:
            config_json = json.load(f)
            model = model_factory(
                config_json["channels"], EModelType(config_json["type"])
            )

            model.fc.load_weights(os.path.join(model_folder, "fc.pt"))
            model.extractor.load_weights(os.path.join(model_folder, "extractor.pt"))

            return model


class AimNet(OsuAiModel):
    """
    Works

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self, channels=CURRENT_STACK_NUM, model_type: EModelType = EModelType.Aim
    ):
        super().__init__(
            channels,
            model_type,
            1024,
            lambda a, b: ResnetFeatureExtractor(a, b),
            lambda a, b: AimRegression(a, b),
        )

    @staticmethod
    def load(model_id: str):
        return OsuAiModel.load(
            model_id, lambda channels, m_type: AimNet(channels, m_type)
        )

    def get_source(self):
        return inspect.getsource(AimNet)


class ActionsNet(OsuAiModel):
    """
    Needs Work

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self, channels=CURRENT_STACK_NUM, model_type: EModelType = EModelType.Actions
    ):
        super().__init__(
            channels,
            model_type,
            1024,
            lambda a, b: ResnetFeatureExtractor(a, b),
            lambda a, b: ClickClassifier(a, b),
        )

    @staticmethod
    def load(model_id: str):
        return OsuAiModel.load(
            model_id, lambda channels, m_type: ActionsNet(channels, m_type)
        )

    def get_source(self):
        return inspect.getsource(ActionsNet)


class CombinedNet(OsuAiModel):
    """
    Needs Work

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self, channels=CURRENT_STACK_NUM, model_type: EModelType = EModelType.Combined
    ):
        super().__init__(
            channels,
            model_type,
            1024,
            lambda a, b: ResnetFeatureExtractor(a, b),
            lambda a, b: CombinedOutput(a, b),
        )

    @staticmethod
    def load(model_id: str):
        return OsuAiModel.load(
            model_id, lambda channels, m_type: CombinedNet(channels, m_type)
        )

    def get_source(self):
        return inspect.getsource(CombinedNet)
