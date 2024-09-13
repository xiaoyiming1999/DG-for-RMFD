import numpy as np
import torch


def mixup_augmentation(inputs, interval, device):
    lamda = np.random.dirichlet(alpha=(1, 1, 1), size=1)
    lamda = torch.as_tensor(lamda).to(device)
    inputs_1 = inputs.narrow(0, 0, interval)
    inputs_2 = inputs.narrow(0, interval, interval)
    inputs_3 = inputs.narrow(0, 2 * interval, interval)
    aug_inputs = lamda[0, 0] * inputs_1 + lamda[0, 1] * inputs_2 + lamda[0, 2] * inputs_3
    aug_inputs = aug_inputs.to(torch.float32)

    return aug_inputs, lamda