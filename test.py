from ai.dataset import OsuDataset
import numpy as np
data = OsuDataset('leia-2-5.41', train_actions=False)

# test_arr = np.array([
#     [1, 2, 3],
#     [1, 2, 3],
#     [1, 2, 3],
#     [1, 2, 3],
#     [1, 2, 3]
# ], dtype=object)

# print(test_arr[:, 0])

print(data[0])
