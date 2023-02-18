import torch
import numpy as np

a = torch.Tensor([[5, 2, 3], [1, 6, 3], [1, 2, 7]])

# torch.Tensor([0, 1, 2]).type(torch.LongTensor)
b = np.arange(3, dtype=np.int32)

c = torch.Tensor([0, 1, 2]).type(torch.LongTensor)

print(b, c, a[b, c])
