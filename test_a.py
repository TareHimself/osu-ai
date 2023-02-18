import time
from ai.eval import AimThread, ActionsThread

# AI = AimThread(
#     model_path=f'D:\Github\osu-ai\models\model_aim_10-things_18-02-23-06-21-23.pt')
AC = ActionsThread(
    model_path=f'D:\Github\osu-ai\models\model_action_10-things_18-02-23-06-27-39.pt')
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    pass
