from torch import optim


class StaticScheduler(object):
    def __init__(self, optimizer):
        pass

    def step(self):
        pass


schedulers = {
    'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
    'MultiStepLR': optim.lr_scheduler.MultiStepLR,
    'OneCycleLR': optim.lr_scheduler.OneCycleLR,
    'StaticScheduler': StaticScheduler
}
