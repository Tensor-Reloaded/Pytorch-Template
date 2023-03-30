import torch.nn as nn
import torch


def init_weights(model, initialization_type):
    match initialization_type:
        case 1:
            xavier_init(model)
        case 2:
            he_init(model)
        case 3:
            selu_init(model)
        case 4:
            orthogonal_init(model)
        case _:
            print(f"Unknown initialization type {initialization_type}")


def xavier_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform(
                m.weight, gain=nn.init.calculate_gain('relu'))


def he_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')


def selu_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))


def orthogonal_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
