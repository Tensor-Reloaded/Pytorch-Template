import logging

from omegaconf import OmegaConf
from torch import optim


def init_scheduler(scheduler_config, optimizer):
    # TODO: Implement more schedulers at once
    (name, parameters) = list(scheduler_config.items())[1]
    if name not in schedulers:
        logging.error(f"Scheduler {name} does not exist!")
        exit()

    parameters = OmegaConf.to_container(parameters, resolve=True)
    parameters = {k: v for k, v in parameters.items() if v is not None}
    parameters["optimizer"] = optimizer
    scheduler = schedulers[name](**parameters)
    return scheduler


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
