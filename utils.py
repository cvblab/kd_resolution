import numpy as np
import torch
import random


def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def min_max_norm(x):

        min_value = torch.min(x.view((x.shape[0], -1)), -1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        max_value = torch.max(x.view((x.shape[0], -1)), -1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        x = (x - min_value) / (max_value - min_value)

        return x
