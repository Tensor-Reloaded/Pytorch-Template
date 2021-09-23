import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch import nn
from .losses import losses
from sklearn.metrics import cohen_kappa_score

class Metric():
    def __init__(self, name, metric_func, index=None, solver_metric = False, aggregator = None):
        self.name = name
        self.metric_func = metric_func
        self.solver_metric = solver_metric
        self.aggregator = aggregator
        self.index = index

        self.batch_accumulator = []
    
    def calculate(self, prediction=None, target=None, solver=None, level = 'epoch'):
        if self.index is not None:
            prediction = prediction[self.index]
            target = target[self.index]
        if self.solver_metric:
            result = self.metric_func(solver)
        else:
            result = self.metric_func(prediction, target)
        
        if self.aggregator is not None:
            self.batch_accumulator.append(result)

        if level == 'batch':
            return result
        elif level == 'epoch':
            if hasattr(self.metric_func, 'class_list') and self.metric_func.class_list is not None:
                target_list = [[row[key] for row in self.batch_accumulator] for key in self.metric_func.class_list]
                result = {}
                for i in range(len(target_list)):
                    aux = torch.Tensor(target_list[i])
                    if self.aggregator == 'mean':
                        result[self.metric_func.class_list[i]] = torch.mean(aux) 
                    elif self.aggregator == 'sum':
                        result[self.metric_func.class_list[i]] = torch.sum(aux) 

            else:
                self.batch_accumulator = torch.Tensor(self.batch_accumulator)
                if self.aggregator == 'mean':
                    result = torch.mean(self.batch_accumulator)
                elif self.aggregator == 'sum':
                    result = torch.sum(self.batch_accumulator)

            self.batch_accumulator = []
            return result

class F1(object):
    def __init__(self, class_list = None, weighted = False):
        self.class_list = class_list
        self.weighted = weighted
        self.precision_func = Precision(class_list, weighted)
        self.recall_func = Recall(class_list, weighted)

    def __call__(self, preds_i, target):
        precision = self.precision_func(preds_i, target)
        recall = self.recall_func(preds_i, target)
        if self.class_list == None:
            return 2 * ((precision * recall) / (precision + recall))
        else:
            return {i: 2 * ((precision[i] * recall[i]) / (precision[i] + recall[i])) for i in self.class_list}


class Accuracy(object):
    def __init__(self, class_list = None, weighted = False, multilabel = False):
        self.class_list = class_list
        self.weighted = weighted
        self.multilabel = multilabel

    def __call__(self, preds_i, target):
        if self.multilabel:
            preds = F.one_hot(preds_i.argmax(-1), target.shape[-1])
            pos_pred, neg_pred = preds == 1, preds == 0
            true_pred, false_pred = target == preds, target != preds
            tp = (true_pred * pos_pred).sum(dim=0)
            fp = (false_pred * pos_pred).sum(dim=0)
        
            tn = (true_pred * neg_pred).sum(dim=0)
            fn = (false_pred * neg_pred).sum(dim=0)
        else:
            preds = preds_i.argmax(-1)
            target = target.argmax(-1)
            true_pred, false_pred = target == preds, target != preds
            tp = (true_pred * true_pred).sum(dim=0)
            fp = (false_pred * true_pred).sum(dim=0)
            tn = (true_pred * false_pred).sum(dim=0)
            fn = (false_pred * false_pred).sum(dim=0)
            
        if self.weighted:
            return ((tp + tn) / ( tp + tn + fp + fn )).nan_to_num(0.0).mean()
        if self.class_list == None:
            tp = tp.sum()
            fp = fp.sum()
            tn = tn.sum()
            fn = fn.sum()
            return ((tp + tn) / ( tp + tn + fp + fn )).nan_to_num(0.0)
        else:
            values = ((tp + tn) / ( tp + tn + fp + fn )).nan_to_num(0.0)

            return {self.class_list[i]:values[i] for i in range(len(self.class_list))}


class Precision(object):
    def __init__(self, class_list = None, weighted = False):
        self.class_list = class_list
        self.weighted = weighted

    def __call__(self, preds_i, target):
        preds = F.one_hot(preds_i.argmax(-1), target.shape[-1])
        true_pred, false_pred = target == preds, target != preds
        pos_pred = preds == 1

        tp = (true_pred * pos_pred).sum(dim=0)
        fp = (false_pred * pos_pred).sum(dim=0)
        if self.weighted:
            return (tp / (tp + fp)).nan_to_num(0.0).mean()
        if self.class_list == None:
            tp = tp.sum()
            fp = fp.sum() 
            return (tp / (tp + fp)).nan_to_num(0.0)
        else:
            values = (tp / (tp + fp)).nan_to_num(0.0)
            return {self.class_list[i]:values[i] for i in range(len(self.class_list))}

class Recall(object):
    def __init__(self, class_list = None, weighted = False):
        self.class_list = class_list
        self.weighted = weighted

    def __call__(self, preds_i, target):
        preds = F.one_hot(preds_i.argmax(-1), target.shape[-1])
        true_pred, false_pred = target == preds, target != preds
        pos_pred = preds == 1
        neg_pred = preds == 0

        tp = (true_pred * pos_pred).sum(dim=0)
        fn = (false_pred * neg_pred).sum(dim=0)
    
        if self.weighted:
            return (tp / (tp + fn)).nan_to_num(0.0).mean()
        if self.class_list == None:
            tp = tp.sum()
            fn = fn.sum()
            return (tp / (tp + fn)).nan_to_num(0.0)
        else:
            values = tp / (tp + fn).nan_to_num(0.0)
            return {self.class_list[i]:values[i] for i in range(len(self.class_list))}


class Identity(object):
    def __init__(self, ):
        pass

    def __call__(self, preds_i, target):
        return preds_i

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
    'F1': {
        'constructor': F1,
        'higher_is_better': True
    },
    'Accuracy': {
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
    'Identity':{
        'constructor': Identity,
        'higher_is_better': False
    },
}

metrics = metrics | losses
