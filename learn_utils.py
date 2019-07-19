'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import random

import numpy as np
import progressbar
import torch
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def get_progress_bar(total):
    format_custom_text = progressbar.FormatCustomText(
        'Loss: %(loss).3f | Acc: %(acc).3f%% (%(c)d/%(t)d)',
        dict(
            loss=0,
            acc=0,
            c=0,
            t=0,
        ),
    )
    prog_bar = progressbar.ProgressBar(0, total, widgets=[
        progressbar.Counter(), ' of {} '.format(total),
        progressbar.Bar(),
        ' ', progressbar.ETA(),
        ' ', format_custom_text
    ])
    return prog_bar, format_custom_text


def update_progress_bar(progress_bar_obj, index=None, loss=None, acc=None, c=None, t=None):
    prog_bar, format_custom_text = progress_bar_obj
    format_custom_text.update_mapping(loss=loss, acc=acc, c=c, t=t)
    prog_bar.update(index)

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def reset_seed(seed):
    """
    Sets seed of all random number generators used to the same seed, given as argument
    WARNING: for full reproducibility of training, torch.backends.cudnn.deterministic = True is also needed!
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def begin_chart(chart_name, x_axis_name):
    print(f'{{"chart":"{chart_name}", "axis": "{x_axis_name}"}}')


def begin_per_epoch_chart(chart_name):
    begin_chart(chart_name, 'Epoch')


def add_chart_point(chart_name, x, y):
    print(f'{{"chart": "{chart_name}", "x":{x}, "y":{y}}}')

def compute_weights_l1_norm(model):
    norm_sum = 0
    for param in model.parameters():
        norm_sum += torch.sum(torch.abs(param))
    return norm_sum