import collections
import sys
import pprint
import argparse
import pickle
import os
import re
from shutil import copyfile

import numpy as np
from skimage import transform
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch_optimizer
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as transforms
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import cohen_kappa_score
from torchvision import datasets

from utils import *
from utils.misc import progress_bar
import models

from timm.data.mixup import Mixup
from timm.data.dataset import ImageDataset
from timm.data.loader import create_loader




@hydra.main(config_path='experiments', config_name='config')
def main(config: DictConfig):
    head, tail = os.path.split(config.save_dir)
    if tail == 'None':
        config.save_dir = head
    else:
        config.save_dir = os.path.join(head, tail)

    save_config_path = "runs/" + config.save_dir
    os.makedirs(save_config_path, exist_ok=True)
    with open(os.path.join(save_config_path, "README.md"), 'w+') as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))

    solver = Solver(config)
    return solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.args = config
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.es = EarlyStopping(patience=self.args.es_patience)
        self.scaler = GradScaler()
        
        # TODO move this into transformations also. Probably do an exception when loading transformations to check if its mixup, if it is set a flag and use the mixup function in the training loop
        mixup_args = {
            'mixup_alpha': 1.,
            'cutmix_alpha': 0.,
            'cutmix_minmax': None,
            'prob': 1.0,
            'switch_prob': 0.,
            'mode': 'batch',
            'label_smoothing': 0,
            'num_classes': self.args.dataset.num_classes
        }

        self.mixup_fn = Mixup(**mixup_args)

        if not self.args.save_dir:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir="runs/" + self.args.save_dir)

        self.train_batch_plot_idx = 0
        self.test_batch_plot_idx = 0

    
    def load_data(self):
        if self.args.dataset.name not in datasets:
            print(f"This dataset is not implemented ({self.args.dataset.name}), go ahead and commit it")
            exit()

        train_transformations = []
        for transformation in self.args.transformations.train:
            if transformation.name not in transformations:
                print(f"This transformation is not implemented ({transformation.name}), go ahead and commit it")
                exit()
            train_transformations.append(transformations[transformation.name](**transformation.parameters))

        test_transformations = []
        for transformation in self.args.transformations.test:
            if transformation.name not in transformations:
                print(f"This transformation is not implemented ({transformation.name}), go ahead and commit it")
                exit()
            test_transformations.append(transformations[transformation.name](**transformation.parameters))

        train_transform = transforms.Compose(train_transformations) if len(train_transformations) > 0 else None
        test_transform = transforms.Compose(test_transformations) if len(test_transformations) > 0 else None

        parameters = OmegaConf.to_container(self.args.dataset.train_loader_params, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        parameters["transform"] = train_transform
        self.train_set = datasets[self.args.dataset.name](**parameters)

        parameters = OmegaConf.to_container(self.args.dataset.test_loader_params, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        parameters["transform"] = train_transform
        self.test_set = datasets[self.args.dataset.name](**parameters)
        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.args.dataset.train_batch_size, shuffle=self.args.dataset.shuffle, num_workers=self.args.dataset.num_workers_train)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set, batch_size=self.args.dataset.test_batch_size, shuffle=False, num_workers=self.args.dataset.num_workers_test)

    def init_model(self):
        if self.cuda:
            self.device = torch.device('cuda' + ":" + str(self.args.cuda_device))
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        parameters = OmegaConf.to_container(self.args.model.parameters, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        try:
            self.model = getattr(models, self.args.model.name)(**parameters)
        except:
            print(f"This model is not implemented ({self.args.model.name}), go ahead and commit it")
            exit()


        self.save_dir = os.path.join(self.args.storage_dir,"model_weights",self.args.save_dir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        if self.args.initialization == 1:
            # xavier init
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform(
                        m.weight, gain=nn.init.calculate_gain('relu'))
        elif self.args.initialization == 2:
            # he initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal(m.weight, mode='fan_in')
        elif self.args.initialization == 3:
            # selu init
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.kernel_size[0] * \
                        m.kernel_size[1] * m.in_channels
                    nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))
                elif isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))
        elif self.args.initialization == 4:
            # orthogonal initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal(m.weight)

        if self.args.initialization_batch_norm:
            # batch norm initialization
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant(m.weight, 1)
                    nn.init.constant(m.bias, 0)

        if len(self.args.load_model) > 0:
            print("Loading model from " + self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model))
        self.model = self.model.to(self.device)

    def init_optimizer(self):
        parameters = OmegaConf.to_container(self.args.optimizer.parameters, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        parameters["params"] = self.model.parameters()
        try:
            self.optimizer = getattr(torch_optimizer, self.args.optimizer.name)(**parameters)
        except Exception as e:
            try:
                self.optimizer = getattr(optim, self.args.optimizer.name)(**parameters)
            except:
                print(f"This optimizer is not implemented ({self.args.optimizer.name}), go ahead and commit it")
                exit()
        if self.args.optimizer.use_lookahead:
            self.optimizer = torch_optimizer.Lookahead(self.optimizer, k=self.args.optimizer.lookahead_k, alpha=self.args.optimizer.lookahead_alpha)

    def init_scheduler(self):
        if self.args.scheduler.name not in schedulers:
            print(f"This loss is not implemented ({self.args.scheduler.name}), go ahead and commit it")
            exit()
            
        parameters = OmegaConf.to_container(self.args.scheduler.parameters, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        parameters["optimizer"] = self.optimizer
        self.scheduler = schedulers[self.args.scheduler.name](**parameters)

    def init_criterion(self):
        if self.args.loss.name not in losses:
            print(f"This loss is not implemented ({self.args.loss.name}), go ahead and commit it")
            exit()

        parameters = OmegaConf.to_container(self.args.loss.parameters, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        self.criterion = losses[self.args.loss.name]['constructor'](**parameters)
        
    def init_metrics(self):
        self.metrics = {
            'train':{
                'batch':[],
                'epoch':[]
            },
            'test':{
                'batch':[],
                'epoch':[]
            },
            'solver':{
                'batch':[],
                'epoch':[]
            },
        }


        for metric in self.args.metrics.train:
            if metric.name not in metrics:
                print(f"This metric is not implemented ({metric.name}), go ahead and commit it")
                exit()

            metric_func = metrics[metric.name]['constructor'](**metric.parameters)
            metric_object = Metric(metric.name, metric_func, solver_metric=False, aggregator=metric.aggregator)
            for level in metric.levels:
                self.metrics['train'][level].append(metric_object)
        
        for metric in self.args.metrics.test:
            if metric.name not in metrics:
                print(f"This metric is not implemented ({metric.name}), go ahead and commit it")
                exit()

            metric_func = metrics[metric.name]['constructor'](**metric.parameters)
            metric_object = Metric(metric.name, metric_func, solver_metric=False, aggregator=metric.aggregator)
            for level in metric.levels:
                self.metrics['test'][level].append(metric_object)   

        for metric in self.args.metrics.solver:
            if metric.name not in metrics:
                print(f"This metric is not implemented ({metric.name}), go ahead and commit it")
                exit()

            metric_func = metrics[metric.name]['constructor'](**metric.parameters)
            metric_object = Metric(metric.name, metric_func, solver_metric=True, aggregator=metric.aggregator)
            for level in metric.levels:
                self.metrics['solver'][level].append(metric_object)


    def train(self):
        print("train:")
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        predictions = []
        targets = []
        for batch_num, (data, target) in enumerate(self.train_loader):
            if isinstance(data,list):
                data = [i.to(self.device) for i in data]
            else:
                data = data.to(self.device)
            if isinstance(target,list):
                target = [i.to(self.device) for i in target]
            else:
                target = target.to(self.device)

            self.optimizer.zero_grad()
            if self.args.half:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    self.scaler.scale(loss).backward()
                    if self.train_batch_plot_idx % self.args.dataset.update_every == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                if self.train_batch_plot_idx % self.args.dataset.update_every == 0:
                    self.optimizer.step()

            predictions.extend(output)
            targets.extend(target)

            metrics_results = {}
            for metric in self.metrics['train']['batch']:
                metrics_results["Train/Batch-"+metric.name] = metric.calculate(output, target, level='batch')

            for metric in self.metrics['solver']['batch']:
                    metrics_results["Solver/Batch-"+metric.name] = metric.calculate(solver = self, level='batch')

            print_metrics(self.writer, metrics_results, self.get_train_batch_plot_idx())

            if self.args.progress_bar:
                progress_bar(batch_num, len(self.train_loader))
            if self.args.scheduler.name == "OneCycleLR":
                self.scheduler.step()

        return torch.stack(predictions), torch.stack(targets)

    def test(self):
        print("test:")
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        predictions = []
        targets = []
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                if isinstance(data,list):
                    data = [i.to(self.device) for i in data]
                else:
                    data = data.to(self.device)
                if isinstance(target,list):
                    target = [i.to(self.device) for i in target]
                else:
                    target = target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                predictions.extend(output)
                targets.extend(target)

                metrics_results = {}
                for metric in self.metrics['test']['batch']:
                    metrics_results["Test/Batch-"+metric.name] = metric.calculate(output, target, level='batch')
                    

                print_metrics(self.writer, metrics_results, self.get_test_batch_plot_idx())

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.test_loader))

        return torch.stack(predictions), torch.stack(targets)


    def save(self, epoch, metric, tag=None):
        if tag != None:
            tag = "_"+tag
        else:
            tag = ""
        model_out_path = os.path.join(self.save_dir, "model_{}_{}{}.pth".format(epoch, metric, tag))
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        if self.args.seed is not None:
            reset_seed(self.args.seed)
        self.load_data()
        self.init_model()
        self.init_optimizer()
        self.init_scheduler()
        self.init_criterion()
        self.init_metrics()

        if self.cuda:
            if self.args.half:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=f"O{self.args.mixpo}",
                                                            patch_torch_functions=True, keep_batchnorm_fp32=True)
        try:
            best_metrics = {}
            higher_is_better = metrics[self.args.optimized_metric.split('/')[-1]]['higher_is_better']
            for epoch in range(1, self.args.epochs + 1):
                print("\n===> epoch: %d/%d" % (epoch, self.args.epochs))
                self.epoch = epoch

                metrics_results = {}

                predictions, targets = self.train()
                for metric in self.metrics['train']['epoch']:
                    metric_name = "Train/"+metric.name 
                    metrics_results[metric_name] = metric.calculate(predictions, targets, level='epoch')
                        
                predictions, targets = self.test()
                for metric in self.metrics['test']['epoch']:
                    metric_name = "Test/"+metric.name 
                    metrics_results[metric_name] = metric.calculate(predictions, targets, level='epoch')
                    
                for metric in self.metrics['solver']['epoch']:
                    metric_name = "Solver/"+metric.name 
                    metrics_results[metric_name] = metric.calculate(solver = self, level='epoch')


                print_metrics(self.writer, metrics_results, self.epoch)

                save_best_metric = False
                if self.args.optimized_metric not in best_metrics:
                    best_metrics[self.args.optimized_metric] = metrics_results[self.args.optimized_metric]
                    save_best_metric = True
                if higher_is_better:
                    if  best_metrics[self.args.optimized_metric] < metrics_results[self.args.optimized_metric]:
                        best_metrics[self.args.optimized_metric] = metrics_results[self.args.optimized_metric]
                        save_best_metric = True
                else:
                    if best_metrics[self.args.optimized_metric] > metrics_results[self.args.optimized_metric]:
                        best_metrics[self.args.optimized_metric] = metrics_results[self.args.optimized_metric]
                        save_best_metric = True


                if save_best_metric:
                    self.save(epoch, best_metrics[self.args.optimized_metric])
                    print("===> BEST "+self.args.optimized_metric+" PERFORMANCE: %.5f" % best_metrics[self.args.optimized_metric])

                if self.args.save_model and epoch % self.args.save_interval == 0:
                    self.save(epoch, 0)

                if self.args.scheduler.name == "MultiStepLR":
                    self.scheduler.step()
                elif self.args.scheduler.name == "ReduceLROnPlateau":
                    self.scheduler.step(metrics_results[self.args.scheduler_metric])
                elif self.args.scheduler.name == "OneCycleLR":
                    pass
                else:
                    self.scheduler.step()

                if self.es.step(metrics_results[self.args.es_metric]):
                    print("Early stopping")
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass

        print("===> BEST "+self.args.optimized_metric+" PERFORMANCE: %.5f" % best_metrics[self.args.optimized_metric])
        files = os.listdir(self.save_dir)
        paths = [os.path.join(self.save_dir, basename) for basename in files if "_0" not in basename]
        if len(paths) > 0:
            src = max(paths, key=os.path.getctime)
            copyfile(src, os.path.join("runs", self.args.save_dir, os.path.basename(src)))

        with open("runs/" + self.args.save_dir + "/README.md", 'a+') as f:
            f.write("\n## "+self.args.optimized_metric+"\n %.5f" % (best_metrics[self.args.optimized_metric]))
        tensorboard_export_dump(self.writer)
        print("Saved best accuracy checkpoint")

        return best_metrics[self.args.optimized_metric]


    def get_train_batch_plot_idx(self):
        self.train_batch_plot_idx += 1
        return self.train_batch_plot_idx - 1

    def get_test_batch_plot_idx(self):
        self.test_batch_plot_idx += 1
        return self.test_batch_plot_idx - 1



if __name__ == '__main__':
    main()
