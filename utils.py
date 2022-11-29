from os import listdir, path, getcwd
from typing import Callable

import torch


def get_models(prefix="") -> list[str]:
    models = listdir(path.normpath(path.join(
        getcwd(), 'models')))

    filtered = []

    for model in models:
        if model.startswith(prefix):
            filtered.append(model)

    return filtered


def get_datasets() -> list[str]:
    pass

def load_model_data(model):
    return torch.load(path.normpath(path.join(
        getcwd(), 'models', model)))

def get_validated_input(prompt="You forgot to put your own prompt", validate_fn=lambda a: len(a.strip()) != 0,
                        conversion_fn=lambda a: a.strip()):
    input_as_str = input(prompt)
    if not validate_fn(input_as_str):
        print("Invalid input, please try again.")
        return get_validated_input(prompt, validate_fn, conversion_fn)

    return conversion_fn(input_as_str)
