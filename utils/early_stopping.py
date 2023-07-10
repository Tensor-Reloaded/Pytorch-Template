import numpy as np


def init_is_better(mode: str, min_delta: float, percentage: bool) -> callable:
    if mode not in {'min', 'max'}:
        raise ValueError('mode ' + mode + ' is unknown!')
    if not percentage:
        if mode == 'min':
            is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            is_better = lambda a, best: a > best + min_delta
    else:
        if mode == 'min':
            multiplier = 1 - min_delta / 100
            is_better = lambda a, best: a < best * multiplier
        if mode == 'max':
            multiplier = 1 + min_delta / 100
            is_better = lambda a, best: a > best * multiplier

    return is_better


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        return False
