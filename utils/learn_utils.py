'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import random
import json
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import hydra

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

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

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def reset_seed(seed):
    """
    Sets seed of all random number generators used to the same seed, given as argument
    WARNING: for full reproducibility of training, torch.backends.cudnn.deterministic = True is also needed!
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_weights_l1_norm(model):
    norm_sum = 0
    for param in model.parameters():
        norm_sum += torch.sum(torch.abs(param))
    return norm_sum

def print_metrics(writer, result, idx):
    for key, value in result.items():
        writer.add_scalar(key, value, idx)


def tensorboard_export_dump(writer):
    assert isinstance(writer, torch.utils.tensorboard.SummaryWriter)
    sns.set()

    tf_files = [] # -> list of paths from writer.log_dir to all files in that directory
    
    for root, dirs, files in os.walk(writer.log_dir):
        for file in files:
            tf_files.append(os.path.join(root,file)) # go over every file recursively in the directory

    for file_id, file in enumerate(tf_files):

        path = os.path.split(file)[0] # determine path to folder in which file lies

        event_acc = EventAccumulator(file)
        event_acc.Reload()
        data = {}

        for tag in sorted(event_acc.Tags()["scalars"]):
            step, value = [], []

            for scalar_event in event_acc.Scalars(tag):
                step.append(scalar_event.step)
                value.append(scalar_event.value)

            data[tag] = (step, value)

        if bool(data):
            with open(path+'/metrics.json', "w") as f:
                json.dump(data, f)
    
    total_metrics = pd.DataFrame(columns=['run', 'tag', 'step', 'value'])
    for root, dirs, files in os.walk(hydra.utils.to_absolute_path('outputs')):
        for file in files:
            metrics = pd.DataFrame(columns=['run', 'tag', 'step', 'value'])
            if file == 'metrics.json':
                data = None
                with open(os.path.join(root,file)) as f:
                    data = json.load(f)
                for key, value in data.items():
                    aux = pd.DataFrame({'step':value[0],'value':value[1]})
                    aux = aux.assign(run=root.split("\\runs\\")[-1])
                    aux = aux.assign(tag=key)
                    metrics = metrics.append(aux, ignore_index=True)

                nr_metrics = len(metrics["tag"].unique())
                fig_nr_columns = int(max(np.ceil(np.sqrt(nr_metrics)),2))
                fig_nr_lines = int(np.ceil(nr_metrics/fig_nr_columns))
                fig, axs = plt.subplots(fig_nr_lines,fig_nr_columns, sharex=False, figsize=(fig_nr_columns*12, fig_nr_lines*12))
                axs = axs.flatten()
                for idx, metric in enumerate(metrics["tag"].unique()):
                    data = metrics.loc[metrics.tag == metric]
                    axs[idx].set_title(metric)
                    axs[idx].set_xlim(0, data.step.max()*1.2)
                    axs[idx].set_xlabel("Batch" if 'Batch' in metric else 'Epoch')
                    axs[idx].set_ylim(data.value.min()*0.8,data.value.max()*1.2)
                    axs[idx].xaxis.set_tick_params(labelbottom=True)
                    ax = sns.lineplot(ax=axs[idx], data=data, x="step", y="value", markers=True)

                fig.savefig(f"{root}/metrics.jpg")
                total_metrics = total_metrics.append(metrics, ignore_index=True)

    total_metrics.to_csv(hydra.utils.to_absolute_path('outputs')+"/total_metrics.csv")

    nr_metrics = len(total_metrics["tag"].unique())
    fig_nr_columns = int(max(np.ceil(np.sqrt(nr_metrics)),2))
    fig_nr_lines = int(np.ceil(nr_metrics/fig_nr_columns))
    fig, axs = plt.subplots(fig_nr_lines,fig_nr_columns, sharex=False, figsize=(fig_nr_columns*12, fig_nr_lines*12))
    axs = axs.flatten()
    for idx, metric in enumerate(total_metrics["tag"].unique()):
        data = total_metrics.loc[total_metrics.tag == metric]
        axs[idx].set_title(metric)
        axs[idx].set_xlim(0,data.step.max()*1.2)
        axs[idx].set_xlabel("Batch" if 'Batch' in metric else 'Epoch')
        axs[idx].set_ylim(data.value.min()*0.8, data.value.max()*1.2)
        axs[idx].xaxis.set_tick_params(labelbottom=True)
        ax = sns.lineplot(ax=axs[idx], data=data, x="step", y="value", hue='run', style='run', markers=True)
    fig.savefig(hydra.utils.to_absolute_path('outputs')+"/total_metrics.jpg")

                
