from capture import start_capture
from utils import get_validated_input
from train import train_new, train_checkpoint
from play import start_play
USER_MAIN_MENU = """
What would you like to do ?
[0] Train a new model
[1] Finetune a model
[2] Collect new data
[3] Test a model
[4] Quit
"""

QUIT_CHOICE = 4


def get_input():
    input_as_str = input(USER_MAIN_MENU).strip()
    if not input_as_str.isnumeric():
        return get_input()

    return int(input_as_str)


def run():
    user_choice = get_validated_input(USER_MAIN_MENU, lambda a: a.strip().isnumeric() and (0 <= int(a.strip()) <= 4), lambda a: int(a.strip()))

    while user_choice != QUIT_CHOICE:
        if user_choice == 0:
            train_new()
        elif user_choice == 1:
            train_checkpoint()
        elif user_choice == 2:
            start_capture()
        elif user_choice == 3:
            start_play()

        user_choice = get_validated_input(USER_MAIN_MENU, lambda a: a.strip().isnumeric(), lambda a: int(a.strip()))
