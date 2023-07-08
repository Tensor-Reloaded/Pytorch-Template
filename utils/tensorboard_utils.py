import json
import os

import hydra
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tensorboard_export_dump(writer, dump_total_metrics=False):
    assert isinstance(writer, torch.utils.tensorboard.SummaryWriter)
    sns.set()

    tf_files = []  # -> list of paths from writer.log_dir to all files in that directory

    for root, dirs, files in os.walk(writer.log_dir):
        for file in files:
            tf_files.append(os.path.join(root, file))  # go over every file recursively in the directory

    for file_id, file in enumerate(tf_files):

        path = os.path.split(file)[0]  # determine path to folder in which file lies

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
            with open(path + '/metrics.json', "w") as f:
                json.dump(data, f)

    total_metrics = pd.DataFrame(columns=['run', 'tag', 'step', 'value'])
    if os.path.exists(hydra.utils.to_absolute_path('outputs')):
        output_dir = hydra.utils.to_absolute_path('outputs')
    elif os.path.exists(hydra.utils.to_absolute_path('multirun')):
        output_dir = hydra.utils.to_absolute_path('multirun')
    else:
        return

    if dump_total_metrics:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                metrics = pd.DataFrame(columns=['run', 'tag', 'step', 'value'])
                if file == 'metrics.json':
                    data = None
                    with open(os.path.join(root, file)) as f:
                        data = json.load(f)
                    for key, value in data.items():
                        aux = pd.DataFrame({'step': value[0], 'value': value[1]})
                        aux = aux.assign(run=root.split("\\runs\\")[-1])
                        aux = aux.assign(tag=key)
                        metrics = metrics.append(aux, ignore_index=True)

                    nr_metrics = len(metrics["tag"].unique())
                    fig_nr_columns = int(max(np.ceil(np.sqrt(nr_metrics)), 2))
                    fig_nr_lines = int(np.ceil(nr_metrics / fig_nr_columns))
                    fig, axs = plt.subplots(fig_nr_lines, fig_nr_columns, sharex=False,
                                            figsize=(fig_nr_columns * 12, fig_nr_lines * 12))
                    axs = axs.flatten()
                    for idx, metric in enumerate(metrics["tag"].unique()):
                        data = metrics.loc[metrics.tag == metric]
                        axs[idx].set_title(metric)
                        axs[idx].set_xlim(0, data.step.max() * 1.2)
                        axs[idx].set_xlabel('Batch' if 'Batch' in metric else 'Epoch')
                        axs[idx].set_ylim(data.value.min() * 0.8, data.value.max() * 1.2)
                        axs[idx].xaxis.set_tick_params(labelbottom=True)
                        ax = sns.lineplot(ax=axs[idx], data=data, x="step", y="value", markers=True)

                    fig.savefig(f"{root}/metrics.jpg")
                    total_metrics = total_metrics.append(metrics, ignore_index=True)

        total_metrics.to_csv(output_dir + "/total_metrics.csv")

        nr_metrics = len(total_metrics["tag"].unique())
        fig_nr_columns = int(max(np.ceil(np.sqrt(nr_metrics)), 2))
        fig_nr_lines = int(np.ceil(nr_metrics / fig_nr_columns))
        fig, axs = plt.subplots(fig_nr_lines, fig_nr_columns, sharex=False,
                                figsize=(fig_nr_columns * 12, fig_nr_lines * 12))
        axs = axs.flatten()
        for idx, metric in enumerate(total_metrics["tag"].unique()):
            data = total_metrics.loc[total_metrics.tag == metric]
            axs[idx].set_title(metric)
            axs[idx].set_xlim(0, data.step.max() * 1.2)
            axs[idx].set_xlabel("Batch" if 'Batch' in metric else 'Epoch')
            axs[idx].set_ylim(data.value.min() * 0.8, data.value.max() * 1.2)
            axs[idx].xaxis.set_tick_params(labelbottom=True)
            ax = sns.lineplot(ax=axs[idx], data=data, x="step", y="value", hue='run', style='run', markers=True)
        fig.savefig(output_dir + "/total_metrics.jpg")
