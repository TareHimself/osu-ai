import torch


pred = torch.Tensor([[[0, 0], [0.5, 0.5]]])

target = torch.Tensor([[[0.5, 0.5], [0.5, 0.5]]])


def compute_element_wise_accuracy(pred, target, radius: int):
    total = 0
    total_correct = 0
    for x in range(len(pred)):
        batch_pred = pred[x]
        batch_target = target[x]
        for y in range(len(batch_pred)):
            coord_pred = batch_pred[y]
            coord_target = batch_target[y]

            if coord_target[0] - radius <= coord_pred[0] <= coord_target[0] + radius and coord_target[1] - radius <= coord_pred[1] <= coord_target[1] + radius:
                total_correct += 1

            total += 1

    return total, total_correct


print(compute_element_wise_accuracy(pred, target, 0.3))
