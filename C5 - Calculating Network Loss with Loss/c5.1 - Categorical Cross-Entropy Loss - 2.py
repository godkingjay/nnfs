import numpy as np

softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])

class_targets = [0, 1, 1]

# for targe_idx, distribution in zip(class_targets, softmax_outputs):
#     print(distribution[targe_idx])

# print(softmax_outputs[[0, 1, 2], class_targets])

# print(softmax_outputs[range(len(softmax_outputs)), class_targets])

ng_log = -np.log(softmax_outputs[
    range(len(softmax_outputs)), class_targets
])

average_loss = np.mean(ng_log)
print(average_loss)
