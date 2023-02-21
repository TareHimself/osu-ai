import time
from ai.eval import AimThread, ActionsThread
from utils import FixedRuntime, OsuSocketServer
# # AI = AimThread(
# #     model_path=f'D:\Github\osu-ai\models\model_aim_10-things_18-02-23-06-21-23.pt')
# AC = ActionsThread(
#     model_path=f'D:\Github\osu-ai\models\model_action_10-things_18-02-23-06-27-39.pt')
socket_server = OsuSocketServer(on_state_updated=lambda a: a)
socket_server.send('save,test,start,0.01')
try:
    while True:
        with FixedRuntime(0.01):
            pass

except KeyboardInterrupt:
    socket_server.send('save,test,stop,0.01')
    pass
