import torch
from functorch import vmap

pred = torch.tensor([
    [1, 2],
    [1, 1],
    [0, 2]
])

correct = torch.tensor([
    [1, 4],
    [1, 0],
    [0, 2]
])
