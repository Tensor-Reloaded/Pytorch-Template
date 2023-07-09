from __future__ import annotations
import logging
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn, autocast
import torch.nn.functional as F

from . import attr_is_valid
from .losses import losses


def register_metrics(training_metrics: Dict[str, Dict[str, Sequence[Metric]]], metric_type: str, metric_level: str,
                     metric_results: dict, **kwargs) -> dict:
    metric_prefix = metric_type.capitalize() + "/" + "Batch-" if metric_level == "batch" else ""
    with autocast(enabled=True, device_type="cpu"):
        # Needed for calculating cross entropy metric, otherwise I receive
        # "log_softmax_lastdim_kernel_impl" not implemented for 'Half'
        for metric in training_metrics[metric_type][metric_level]:
            metric_name = metric_prefix + metric.name
            result = metric.calculate(level=metric_level, **kwargs)
            if type(result) is dict:
                for each_key in result.keys():
                    metric_results[metric_name + f"_{each_key}"] = result[each_key]
            else:
                metric_results[metric_name] = result
        return metric_results


def init_metric(metric_args, metric_name: str, solver_metric: bool, training_metrics: dict):
    full_metric_name = f"{metric_name}_metrics"
    if hasattr(metric_args, full_metric_name):
        for name, args in getattr(metric_args, full_metric_name).items():
            if name not in metrics:
                logging.error(f"Metric {name} from {full_metric_name} does not exist!")
                exit()

            if args.parameters is None:
                args.parameters = {}

            if name in losses:
                metric_func = LossWrapper(metrics[name]['constructor'](**args.parameters))
            else:
                metric_func = metrics[name]['constructor'](**args.parameters)
            metric_object = Metric(name, metric_func, solver_metric=solver_metric, aggregator=args.aggregator)

            for level in args.levels:
                training_metrics[metric_name][level].append(metric_object)


def init_metrics(metric_args):
    training_metrics = {
        'train': {
            'batch': [],
            'epoch': []
        },
        'val': {
            'batch': [],
            'epoch': []
        },
        'test': {
            'batch': [],
            'epoch': []
        },
        'solver': {
            'batch': [],
            'epoch': []
        },
    }

    init_metric(metric_args, "train", False, training_metrics)
    init_metric(metric_args, "val", False, training_metrics)
    init_metric(metric_args, "test", False, training_metrics)
    init_metric(metric_args, "solver", True, training_metrics)

    return training_metrics


class LossWrapper:
    def __init__(self, loss_func):
        self.loss_func = loss_func

    def __call__(self, **kwargs):
        return self.loss_func(kwargs["prediction"], kwargs["target"])


class Metric:
    def __init__(self, name, metric_func, solver_metric, aggregator):
        self.name = name
        self.metric_func = metric_func
        self.solver_metric = solver_metric

        if aggregator is None:
            self.aggregator = None
        elif aggregator == "mean":
            self.aggregator = lambda x: torch.mean(torch.Tensor(x))
        elif aggregator == "sum":
            self.aggregator = lambda x: torch.sum(torch.Tensor(x))
        else:
            raise RuntimeError("Unknown aggregator {aggregator}")

        if self.aggregator is not None:
            self.batch_accumulator = []

    def calculate(self, **kwargs):
        level = kwargs.pop("level")
        result = self.metric_func(**kwargs)

        if self.aggregator is None:
            return result

        self.batch_accumulator.append(result)
        if level == "epoch":
            result = self.accumulate()

        return result

    def accumulate(self):
        if attr_is_valid(self.metric_func, "class_list"):
            target_list = [[row[key] for row in self.batch_accumulator] for key in self.metric_func.class_list]
            result = {}
            for i in range(len(target_list)):
                result[self.metric_func.class_list[i]] = self.aggregator(target_list[i])

        else:
            result = self.aggregator(self.batch_accumulator)

        self.batch_accumulator = []
        return result

    def __call__(self, **kwargs):
        return self.calculate(kwargs)


class F1(object):
    def __init__(self, class_list=None, weighted=False):
        self.class_list = class_list
        self.weighted = weighted
        self.precision_func = Precision(class_list, weighted)
        self.recall_func = Recall(class_list, weighted)

    def __call__(self, **kwargs):
        prediction, target = kwargs["prediction"], kwargs["target"]
        precision = self.precision_func(prediction, target)
        recall = self.recall_func(prediction, target)

        if self.class_list is None:
            return 2 * ((precision * recall) / (precision + recall))
        else:
            return {i: 2 * ((precision[i] * recall[i]) / (precision[i] + recall[i])) for i in self.class_list}


# TODO: Use library accuracy
class Accuracy(object):
    def __init__(self, class_list=None, weighted=False, multilabel=False, topk=1):
        self.class_list = class_list
        self.weighted = weighted
        self.multilabel = multilabel
        self.topk = topk

    def __call__(self, **kwargs):
        prediction, target = kwargs["prediction"], kwargs["target"]
        # TODO: Use library code
        if self.topk > 1:
            with torch.no_grad():
                batch_size = target.size(0)

                _, y_pred = prediction.topk(k=self.topk, dim=1)
                y_pred = y_pred.t()
                target_reshaped = target.argmax(-1).view(1, -1).expand_as(y_pred)
                correct = (y_pred == target_reshaped)

                ind_which_topk_matched_truth = correct[:self.topk]
                flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
                tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)
                topk_acc = tot_correct_topk / batch_size
                return topk_acc
        if self.multilabel:
            preds = F.one_hot(prediction.argmax(-1), target.shape[-1])
            pos_pred, neg_pred = preds == 1, preds == 0
            true_pred, false_pred = target == preds, target != preds
            tp = (true_pred * pos_pred).sum(dim=0)
            fp = (false_pred * pos_pred).sum(dim=0)

            tn = (true_pred * neg_pred).sum(dim=0)
            fn = (false_pred * neg_pred).sum(dim=0)
        else:
            preds = prediction.argmax(-1)
            target = target.argmax(-1)
            true_pred, false_pred = (target == preds, target != preds)
            tp = (true_pred * true_pred).sum(dim=0)
            fp = (false_pred * true_pred).sum(dim=0)
            tn = (true_pred * false_pred).sum(dim=0)
            fn = (false_pred * false_pred).sum(dim=0)

        if self.weighted:
            return ((tp + tn) / (tp + tn + fp + fn)).nan_to_num(0.0).mean()
        if self.class_list is None:
            tp = tp.sum()
            fp = fp.sum()
            tn = tn.sum()
            fn = fn.sum()
            return ((tp + tn) / (tp + tn + fp + fn)).nan_to_num(0.0)
        else:
            values = ((tp + tn) / (tp + tn + fp + fn)).nan_to_num(0.0)

            return {self.class_list[i]: values[i] for i in range(len(self.class_list))}


# TODO: Recall and precision could be calculated together, at the same time, and it would be faster
class Precision(object):
    def __init__(self, class_list=None, weighted=False):
        self.class_list = class_list
        self.weighted = weighted

    def __call__(self, **kwargs):
        prediction = kwargs["prediction"], target = kwargs["target"]
        # TODO: Use library code
        preds = F.one_hot(prediction.argmax(-1), target.shape[-1])
        true_pred, false_pred = target == preds, target != preds
        pos_pred = preds == 1

        tp = (true_pred * pos_pred).sum(dim=0)
        fp = (false_pred * pos_pred).sum(dim=0)
        if self.weighted:
            return (tp / (tp + fp)).nan_to_num(0.0).mean()
        if self.class_list is None:
            tp = tp.sum()
            fp = fp.sum()
            return (tp / (tp + fp)).nan_to_num(0.0)
        else:
            values = (tp / (tp + fp)).nan_to_num(0.0)
            return {self.class_list[i]: values[i] for i in range(len(self.class_list))}


class Recall(object):
    def __init__(self, class_list=None, weighted=False):
        self.class_list = class_list
        self.weighted = weighted

    def __call__(self, **kwargs):
        prediction, target = kwargs["prediction"], kwargs["target"]
        # TODO: Use library code
        preds = F.one_hot(prediction.argmax(-1), target.shape[-1])
        true_pred, false_pred = target == preds, target != preds
        pos_pred = preds == 1
        neg_pred = preds == 0

        tp = (true_pred * pos_pred).sum(dim=0)
        fn = (false_pred * neg_pred).sum(dim=0)

        if self.weighted:
            return (tp / (tp + fn)).nan_to_num(0.0).mean()
        if self.class_list is None:
            tp = tp.sum()
            fn = fn.sum()
            return (tp / (tp + fn)).nan_to_num(0.0)
        else:
            values = tp / (tp + fn).nan_to_num(0.0)
            return {self.class_list[i]: values[i] for i in range(len(self.class_list))}


class Model_Norm(object):
    def __init__(self, norm_type=2):
        self.norm_type = norm_type

    def __call__(self, **kwargs):
        norm = 0.0
        for param in kwargs["solver"].model.parameters():
            norm += torch.norm(input=param, p=self.norm_type, dtype=torch.float)
        return norm


class Learning_Rate(object):
    def __call__(self, **kwargs):
        return kwargs["solver"].optimizer.param_groups[0]['lr']


class Average_Grad_Count(object):
    def __call__(self, **kwargs):  # TODO: Checl
        return kwargs["solver"].odba.avg_desired_batch_size


class Epoch_Utilization(object):
    def __call__(self, **kwargs):  # TODO: Check
        return kwargs["solver"].dba.bs_usage


class Real_Epoch_Count(object):
    def __init__(self):
        self.counter = 0.0

    def __call__(self, **kwargs):  # TODO: Check
        self.counter += kwargs["solver"].dba.bs_usage
        return self.counter


class Batch_Size(object):
    def __init__(self, mode='train'):
        self.display_name = 'Batch Size'
        self.higher_is_better = True
        self.mode = mode

    def __call__(self, **kwargs):
        solver = kwargs["solver"]
        if self.mode == 'train':
            if solver.train_loader.batch_sampler is not None:
                return solver.train_loader.batch_sampler.batch_size
            return solver.train_loader.dataset.dataset.batch_size
        elif self.mode == 'val':
            return solver.val_loader.batch_sampler.batch_size
        elif self.mode == 'test':
            return solver.test_loader.batch_sampler.batch_size
        elif self.mode == 'infer':
            return solver.infer_loader.batch_sampler.batch_size
        else:
            raise ValueError('mode should be train, val, test or infer')


class Perplexity:
    def __call__(self, **kwargs):
        if "loss" in kwargs:
            return np.exp(kwargs["loss"])
        return np.exp(nn.CrossEntropyLoss()(kwargs["prediction"], kwargs["target"]))


metrics = {
    'F1': {
        'constructor': F1,
        'higher_is_better': True
    },
    'Accuracy': {
        'constructor': Accuracy,
        'higher_is_better': True
    },
    'Top K Accuracy': {
        'constructor': Accuracy,
        'higher_is_better': True
    },
    'Precision': {
        'constructor': Precision,
        'higher_is_better': True
    },
    'Recall': {
        'constructor': Recall,
        'higher_is_better': True
    },
    'Model Norm': {
        'constructor': Model_Norm,
        'higher_is_better': False
    },
    'Learning Rate': {
        'constructor': Learning_Rate,
        'higher_is_better': None
    },
    'Average Grad Count': {
        'constructor': Average_Grad_Count,
        'higher_is_better': None
    },
    'Epoch Utilization': {
        'constructor': Epoch_Utilization,
        'higher_is_better': None
    },
    'Real Epoch Count': {
        'constructor': Real_Epoch_Count,
        'higher_is_better': None
    },
    'Batch Size': {
        'constructor': Batch_Size,
        'higher_is_better': None
    },
    'Perplexity': {
        'constructor': Perplexity,
        'higher_is_better': None
    },
}

metrics = metrics | losses
