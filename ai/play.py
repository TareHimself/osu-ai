from ai.utils import FixedRuntime, get_models, get_validated_input, EModelType
from ai.eval import ActionsThread, AimThread, CombinedThread
import traceback


def start_play():
    try:
        action_models = get_models(EModelType.Actions)

        aim_models = get_models(EModelType.Aim)

        combined_models = get_models(EModelType.Combined)

        user_choice = get_validated_input(f"""What type of model would you like to test?
    [0] Aim Model | {len(aim_models)} Available
    [1] Actions Model | {len(action_models)} Available
    [2] Combined Model | {len(combined_models)} Available
""", lambda a: a.strip().isnumeric() and (0 <= int(a.strip()) <= 2), lambda a: int(a.strip()))

        active_model = None
        if user_choice == 0:
            prompt = "What aim model would you like to use?\n"
            for i in range(len(aim_models)):
                prompt += f"    [{i}] {aim_models[i]}\n"

            model_index = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                    0 <= int(a.strip()) < len(aim_models)), lambda a: int(a.strip()))

            active_model = AimThread(model_id=aim_models[model_index]['id'])

        elif user_choice == 1:
            prompt = "What actions model would you like to use?\n"
            for i in range(len(action_models)):
                prompt += f"    [{i}] {action_models[i]}\n"

            model_index = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                    0 <= int(a.strip()) < len(action_models)), lambda a: int(a.strip()))

            active_model = ActionsThread(
                model_id=action_models[model_index]['id'])
        else:
            prompt = "What combined model would you like to use?\n"
            for i in range(len(combined_models)):
                prompt += f"    [{i}] {combined_models[i]}\n"

            model_index = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                    0 <= int(a.strip()) < len(combined_models)), lambda a: int(a.strip()))

            active_model = CombinedThread(
                model_id=combined_models[model_index]['id'])

        try:
            while True:
                with FixedRuntime(2):
                    pass

        except KeyboardInterrupt as e:
            if active_model is not None:
                active_model.kill()
    except Exception as e:
        traceback.print_exc()
