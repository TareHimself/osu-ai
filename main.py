from ai import start_play, start_train, start_capture, get_validated_input

USER_MAIN_MENU = """What would you like to do ?
    [0] Train
    [1] Collect data
    [2] Test
    [3] Quit
"""

QUIT_CHOICE = 3


def get_input():
    input_as_str = input(USER_MAIN_MENU).strip()
    if not input_as_str.isnumeric():
        return get_input()

    return int(input_as_str)


def run():
    get_input_params = [USER_MAIN_MENU, lambda a: a.strip().isnumeric() and (
        0 <= int(a.strip()) <= 3), lambda a: int(a.strip())]

    user_choice = get_validated_input(*get_input_params)

    while user_choice != QUIT_CHOICE:
        if user_choice == 0:
            start_train()
        elif user_choice == 1:
            start_capture()
        elif user_choice == 2:
            start_play()

        user_choice = get_validated_input(*get_input_params)


run()
