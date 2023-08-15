
if __name__ == '__main__':

    from ai.utils import get_validated_input

    USER_MAIN_MENU = """What would you like to do ?
    [0] Train or finetune a model
    [1] Convert a video and json into a dataset
    [2] Test a model
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
                from ai.train import start_train
                start_train()
            elif user_choice == 1:
                from ai.convert import start_convert
                start_convert()
            elif user_choice == 2:
                from ai.play import start_play
                start_play()

            user_choice = get_validated_input(*get_input_params)
    
    run()
