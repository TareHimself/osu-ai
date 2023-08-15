from ai.converter import ReplayConverter
import os

# for dataset in os.listdir("./to_convert"):
#     ReplayConverter(dataset,danser_video=os.path.join('./to_convert',dataset,f"{dataset}.mkv"),replay_json=os.path.join('./to_convert',dataset,f"{dataset}.json"),save_dir="./",num_readers=3,debug=False)
# # ReplayConverter("test",danser_video="sawai miku - colorful (wo cursor).mkv",replay_json="sawai miku - colorful (wo cursor).json",save_dir="./",num_readers=1,video_fps=60,frame_interval_ms=50)

# import torch

# a = torch.Tensor([[1,1,1,0]])
# b = torch.Tensor([[1,1,1,1]])

# def get_acc(pred: torch.Tensor,truth: torch.Tensor,thresh: int = 60,is_combined=False):
#     pred = pred.detach().clone()
#     truth = truth.detach().clone()

#     pred[:,0] *= 1920
#     pred[:,1] *= 1080
#     truth[:,0] *= 1920
#     truth[:,1] *= 1080

#     diff = (pred[:,:-2] - truth[:,:-2]) if is_combined else pred - truth

#     dist = torch.sqrt((diff ** 2).sum(dim=1))

#     dist[dist < thresh] = 1

#     dist[dist >= thresh] = 0

#     if not is_combined:
#         return dist.mean().item()

#     pred_keys = pred[:,2:]
#     truth_keys = truth[:,2:]

#     pred_keys[pred_keys >= 0.5] = 1
#     truth_keys[truth_keys >= 0.5] = 1
#     pred_keys[pred_keys < 0.5] = 0
#     truth_keys[truth_keys < 0.5] = 0

#     return (dist.mean().item() + torch.all(pred_keys == truth_keys,dim=1).float().mean().item()) / 2


# print(a,'\n',b,'\n',get_acc(a,b,is_combined=True))


ReplayConverter("Rightfully 8", "Rightfully 8.mkv",
                "Rightfully 8.json", max_in_memory=5000, save_dir="./",
                num_writers=1, debug=True)
