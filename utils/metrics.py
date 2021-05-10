import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch import nn
from .losses import losses
from sklearn.metrics import cohen_kappa_score

class Metric():
    def __init__(self, name, metric_func, solver_metric = False, aggregator = None):
        self.name = name
        self.metric_func = metric_func
        self.solver_metric = solver_metric
        self.aggregator = aggregator

        self.batch_accumulator = []
    
    def calculate(self, prediction=None, target=None, solver=None, level = 'epoch'):
        if self.solver_metric:
            result = self.metric_func(solver)
        else:
            result = self.metric_func(prediction, target)
        
        if self.aggregator is not None:
            self.batch_accumulator.append(result)

        if level == 'batch':
            return result
        elif level == 'epoch':
            self.batch_accumulator = torch.Tensor(self.batch_accumulator)
            if self.aggregator == 'mean':
                result = torch.mean(self.batch_accumulator)
            elif self.aggregator == 'sum':
                result = torch.sum(self.batch_accumulator)

            self.batch_accumulator = []
            return result


class Accuracy(object):
    def __init__(self):
        pass

    def __call__(self, prediction, target):
        prediction = prediction.argmax(dim=-1)
        target = target.argmax(dim=-1)
        correct = (prediction == target).sum().item()
        return correct / prediction.size(0)

class Cohen_Kappa_Score(object):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
    def __init__(self, weights = None, sample_weight = None, labels = None):
        self.weights = weights
        self.sample_weight = sample_weight
        self.labels = labels
    def __call__(self, prediction, target):
        return cohen_kappa_score(prediction.detach().argmax(dim=-1).cpu().long().numpy(), target.detach().argmax(dim=-1).cpu().long().numpy(), weights=self.weights, sample_weight=self.sample_weight, labels=self.labels)

class Model_Norm(object):
    def __init__(self, norm_type=2):
        self.norm_type = norm_type

    def __call__(self, solver):
        norm = 0.0
        for param in solver.model.parameters():
            norm += torch.norm(input=param, p=self.norm_type, dtype=torch.float)
        return norm

class Learning_Rate(object):
    def __init__(self):
        pass

    def __call__(self, solver):
        return solver.optimizer.param_groups[0]['lr']

metrics = {
    'Accuracy': {
        'constructor': Accuracy,
        'higher_is_better': True
    },
    'Cohen Kappa Score': {
        'constructor': Cohen_Kappa_Score,
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
}

metrics = metrics | losses