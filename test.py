import torch

DISTANCE_CHECK = torch.zeros((2))
MAX_ERROR = 0.05
a = torch.Tensor([[.71, .81], [.6, .9]])

b = torch.Tensor([[.715, .815], [.69, .93]])
# .type(torch.LongTensor)
print(((a - b).absolute() <
      MAX_ERROR), (((a - b).absolute() <
                    MAX_ERROR).sum(dim=1) - 1).clamp(torch.Tensor([0]), torch.Tensor([1])).sum())

print(((torch.pairwise_distance(a, b) - MAX_ERROR)
      < 0))
