import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def apply_if_exists(args, attr: str, op: callable):
    if hasattr(args, attr):
        return op(getattr(args, attr))
    return None


def attr_is_valid(obj: object, attr: str):
    return hasattr(obj, attr) and getattr(obj, attr)


def maybe_reset_seed(seed: int) -> None:
    """
    Sets seed of all random number generators used to the same seed, given as argument
    WARNING: for full reproducibility of training, torch.backends.cudnn.deterministic = True is also needed!
    """
    if seed is None:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # This should be set only if the input size does not changes that much
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_metrics(writer: SummaryWriter, result: dict, idx: int) -> None:
    for key, value in result.items():
        writer.add_scalar(key, value, idx)
