import torch
from torch import nn
from torch import optim


schedulers = {
    'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
    'MultiStepLR': optim.lr_scheduler.MultiStepLR,
    'OneCycleLR': optim.lr_scheduler.OneCycleLR,
}