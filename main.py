import collections
import sys
import pprint
import argparse
import pickle
import os
import re
from shutil import copyfile
import pandas as pd

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
        self.infer_loader = None
        self.es = EarlyStopping(patience=self.args.es_patience)
        self.scaler = GradScaler(enabled=self.args.half)


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
        train_cache_index = 0
        train_data_transformations = []
        for idx, transformation in enumerate(self.args.transformations.train.data):
            if transformation.name not in transformations:
                print(f"This transformation is not implemented ({transformation.name}), go ahead and commit it")
                exit()
            if hasattr(transformation, 'cache_point'):
                train_cache_index = idx+1
            train_data_transformations.append(transformations[transformation.name](**transformation.parameters))

        train_target_transformations = []
        for transformation in self.args.transformations.train.target:
            if transformation.name not in transformations:
                print(f"This transformation is not implemented ({transformation.name}), go ahead and commit it")
                exit()
            train_target_transformations.append(transformations[transformation.name](**transformation.parameters))

        train_both_transformations = []
        for transformation in self.args.transformations.train.both:
            if transformation.name not in transformations:
                print(f"This transformation is not implemented ({transformation.name}), go ahead and commit it")
                exit()
            train_both_transformations.append(transformations[transformation.name](**transformation.parameters))

        train_output_transformations = []
        for transformation in self.args.transformations.train.output:
            if transformation.name not in transformations:
                print(f"This transformation is not implemented ({transformation.name}), go ahead and commit it")
                exit()
            train_output_transformations.append(transformations[transformation.name](**transformation.parameters))

        train_data_transform = transforms.Compose(train_data_transformations) if len(train_data_transformations) > 0 else None
        train_target_transform = transforms.Compose(train_target_transformations) if len(train_target_transformations) > 0 else None
        train_both_transform = transforms.Compose(train_both_transformations) if len(train_both_transformations) > 0 else None
        self.train_output_transform = transforms.Compose(train_output_transformations) if len(train_output_transformations) > 0 else None

        test_cache_index = 0
        test_data_transformations = []
        for idx, transformation in enumerate(self.args.transformations.test.data):
            if transformation.name not in transformations:
                print(f"This transformation is not implemented ({transformation.name}), go ahead and commit it")
                exit()
            if hasattr(transformation, 'cache_point'):
                test_cache_index = idx+1
            test_data_transformations.append(transformations[transformation.name](**transformation.parameters))

        test_target_transformations = []
        for transformation in self.args.transformations.test.target:
            if transformation.name not in transformations:
                print(f"This transformation is not implemented ({transformation.name}), go ahead and commit it")
                exit()
            test_target_transformations.append(transformations[transformation.name](**transformation.parameters))

        test_both_transformations = []
        for transformation in self.args.transformations.test.both:
            if transformation.name not in transformations:
                print(f"This transformation is not implemented ({transformation.name}), go ahead and commit it")
                exit()
            test_both_transformations.append(transformations[transformation.name](**transformation.parameters))

        test_output_transformations = []
        for transformation in self.args.transformations.test.output:
            if transformation.name not in transformations:
                print(f"This transformation is not implemented ({transformation.name}), go ahead and commit it")
                exit()
            test_output_transformations.append(transformations[transformation.name](**transformation.parameters))

        test_data_transform = transforms.Compose(test_data_transformations) if len(test_data_transformations) > 0 else None
        test_target_transform = transforms.Compose(test_target_transformations) if len(test_target_transformations) > 0 else None
        test_both_transform = transforms.Compose(test_both_transformations) if len(test_both_transformations) > 0 else None
        self.test_output_transform = transforms.Compose(test_output_transformations) if len(test_output_transformations) > 0 else None


        parameters = OmegaConf.to_container(self.args.dataset.train_loader_params, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        if self.args.dataset.name in ['CIFAR-10','CIFAR-100','ImageNet2012']:
            parameters["transform"] = train_data_transform
            parameters["target_transform"] = train_target_transform
        else:
            parameters["data_transform"] = train_data_transform
            parameters["target_transform"] = train_target_transform
            parameters["both_transform"] = train_both_transform
            parameters['cache_index'] = train_cache_index
        self.train_set = datasets[self.args.dataset.name](**parameters)

        parameters = OmegaConf.to_container(self.args.dataset.test_loader_params, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        if self.args.dataset.name in ['CIFAR-10','CIFAR-100','ImageNet2012']:
            parameters["transform"] = test_data_transform
            parameters["target_transform"] = test_target_transform
        else:
            parameters["data_transform"] = test_data_transform
            parameters["target_transform"] = test_target_transform
            parameters["both_transform"] = test_both_transform
            parameters['cache_index'] = test_cache_index
        self.test_set = datasets[self.args.dataset.name](**parameters)

        if hasattr(self.args.dataset, 'mixup_args') and self.args.dataset.mixup_args != None:
            collate_fn = FastCollateMixup(**self.args.dataset.mixup_args)
        else:
            collate_fn = None

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.args.dataset.train_batch_size, shuffle=self.args.dataset.shuffle, num_workers=self.args.dataset.num_workers_train, collate_fn=collate_fn, drop_last=True, persistent_workers=self.args.dataset.num_workers_train>0)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set, batch_size=self.args.dataset.test_batch_size, shuffle=False, num_workers=self.args.dataset.num_workers_test, persistent_workers=self.args.dataset.num_workers_test>0)
        if self.args.infer_only is True:
            parameters = OmegaConf.to_container(self.args.dataset.infer_loader_params, resolve=True)
            parameters = {k: v for k, v in parameters.items() if v is not None}
            parameters["data_transform"] = test_data_transform
            parameters["target_transform"] = test_target_transform
            parameters["both_transform"] = test_both_transform
            self.infer_set = datasets[self.args.dataset.name](**parameters)
            self.infer_loader = torch.utils.data.DataLoader(dataset=self.infer_set, batch_size=self.args.dataset.test_batch_size, shuffle=False, num_workers=self.args.dataset.num_workers_test, persistent_workers=self.args.dataset.num_workers_test>0)


    def init_model(self):
        if self.cuda:
            self.device = torch.device('cuda' + ":" + str(self.args.cuda_device))
            cudnn.benchmark = True

            # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
            torch.backends.cuda.matmul.allow_tf32 = True

            # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = torch.device('cpu')

        parameters = OmegaConf.to_container(self.args.model.parameters, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        try:
            self.model = getattr(models, self.args.model.name)
        except:
            print(f"This model is not implemented ({self.args.model.name}), go ahead and commit it")
            exit()
        self.model = self.model(**parameters)


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
                    nn.init.orthogonal_(m.weight)

        if self.args.initialization_batch_norm:
            # batch norm initialization
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if len(self.args.load_model) > 0:
            print("Loading model from " + self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model))

            # for param in self.model.parameters():
            #     param.requires_grad = True
            # for param in self.model.patch_embed.parameters():
            #     param.requires_grad = True
            # for param in self.model.norm.parameters():
            #     param.requires_grad = True
            # for param in self.model.avgpool.parameters():
            #     param.requires_grad = True
            # for param in self.model.head.parameters():
            #     param.requires_grad = True


        self.model = self.model.to(self.device)

    def init_optimizer(self):
        parameters = OmegaConf.to_container(self.args.optimizer.parameters, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        parameters["params"] = self.model.parameters()

        try:
            self.optimizer = getattr(torch_optimizer, self.args.optimizer.name)
        except Exception as e:
            try:
                self.optimizer = getattr(optim, self.args.optimizer.name)
            except:
                print(f"This optimizer is not implemented ({self.args.optimizer.name}), go ahead and commit it")
                exit()

        self.optimizer = self.optimizer(**parameters)
        
        if self.args.optimizer.use_SAM:
            self.optimizer = optimizers['SAM'](base_optimizer=self.optimizer,rho=self.args.optimizer.SAM_rho)
        
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


    def disable_bn(self):
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._NormBase) or isinstance(module, nn.LayerNorm):
                module.eval()

    def enable_bn(self):
        self.model.train()

    def train(self):
        print("train:")
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        accumulation_data = []
        accumulation_target = []

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

            if self.args.optimizer.use_SAM:
                accumulation_data.append(data)
                accumulation_target.append(target)

            while True: 
                with autocast(enabled=self.args.half):
                    output = self.model(data)
                    if self.train_output_transform is not None:
                        output = self.train_output_transform(output)
                    loss = self.criterion(output, target)
                    loss = loss / self.args.dataset.update_every

                if self.args.optimizer.grad_penalty is not None and self.args.optimizer.grad_penalty > 0.0:
                    # Creates gradients
                    scaled_grad_params = torch.autograd.grad(outputs=self.scaler.scale(loss), inputs=self.model.parameters(), create_graph=True)

                    #Creates unscaled grad_params before computing the penalty. scaled_grad_params are
                    # not owned by any optimizer, so ordinary division is used instead of scaler.unscale_:
                    inv_scale = 1./self.scaler.get_scale()
                    grad_params = [p * inv_scale for p in scaled_grad_params]

                    # Computes the penalty term and adds it to the loss
                    with autocast():
                        grad_norm = 0
                        for grad in grad_params:
                            grad_norm += grad.pow(2).sum()
                        grad_norm = grad_norm.sqrt()
                        loss = loss + (grad_norm * self.args.optimizer.grad_penalty)

                self.scaler.scale(loss).backward()

                def sam_closure():
                    self.disable_bn()
                    for i in range(len(accumulation_data)):
                        with autocast(enabled=self.args.half):
                            output = self.model(accumulation_data[i])
                            if self.train_output_transform is not None:
                                output = self.train_output_transform(output)
                            loss = self.criterion(output, accumulation_target[i])
                            loss = loss / self.args.dataset.update_every
                        
                        if self.args.optimizer.grad_penalty is not None and self.args.optimizer.grad_penalty is not False and self.args.optimizer.grad_penalty > 0.0:
                            # Creates gradients
                            scaled_grad_params = torch.autograd.grad(outputs=self.scaler.scale(loss), inputs=self.model.parameters(), create_graph=True)

                            #Creates unscaled grad_params before computing the penalty. scaled_grad_params are
                            # not owned by any optimizer, so ordinary division is used instead of scaler.unscale_:
                            inv_scale = 1./self.scaler.get_scale()
                            grad_params = [p * inv_scale for p in scaled_grad_params]

                            # Computes the penalty term and adds it to the loss
                            with autocast():
                                grad_norm = 0
                                for grad in grad_params:
                                    grad_norm += grad.pow(2).sum()
                                grad_norm = grad_norm.sqrt()
                                loss = loss + (grad_norm * self.args.optimizer.grad_penalty)

                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optimizer.max_norm)
                    self.enable_bn()
                    

                if self.args.optimizer.batch_replay:
                    found_inf = False
                    for _, param in self.model.named_parameters():
                        if  param.grad.isnan().any() or param.grad.isinf().any():
                            found_inf = True
                            break
                    if found_inf:
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        if type(self.args.optimizer.batch_replay) == int or type(self.args.optimizer.batch_replay) == float: 
                            self.args.optimizer.batch_replay -= 1
                    else:
                        break
                else:
                    break

            if self.train_batch_plot_idx % self.args.dataset.update_every == 0:
                self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optimizer.max_norm)
                self.scaler.step(self.optimizer, closure=sam_closure if self.args.optimizer.use_SAM else None)
                self.scaler.update()

                self.optimizer.zero_grad()

                accumulation_data = []
                accumulation_target = []

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

                with autocast(enabled=self.args.half):
                    output = self.model(data)
                    if self.test_output_transform is not None:
                        output = self.test_output_transform(output)
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

    def infer(self):
        print("infer:")
        self.model.eval()

        predictions = []
        filenames = []
        with torch.no_grad():
            for batch_num, (filename, data) in enumerate(self.infer_loader):
                if isinstance(data, list):
                    data = [i.to(self.device) for i in data]
                else:
                    data = data.to(self.device)

                with autocast(enabled=self.args.half):
                    output = self.model(data)
                    if self.test_output_transform is not None:
                        output = self.test_output_transform(output)

                predictions.extend(output)
                filenames.extend(filename)

        return filenames, torch.stack(predictions)


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

        try:
            if self.args.infer_only == True:
                filenames, predictions = self.infer() # If its the "separated" dataset, we need to average the scores of the 2/3 different projections 
                predictions = predictions.argmax(-1)+1
                save_path = os.path.join(self.save_dir, "predictions.csv")
                pd.DataFrame({'Patient': filenames, 'Class': predictions.cpu().numpy()}).to_csv(save_path, header=False, index=False)
                exit()

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

                if self.epoch % self.args.test_every == 0:
                    predictions, targets = self.test()
                    for metric in self.metrics['test']['epoch']:
                        metric_name = "Test/"+metric.name
                        metrics_results[metric_name] = metric.calculate(predictions, targets, level='epoch')

                for metric in self.metrics['solver']['epoch']:
                    metric_name = "Solver/"+metric.name
                    metrics_results[metric_name] = metric.calculate(solver = self, level='epoch')


                print_metrics(self.writer, metrics_results, self.epoch)

                
                if self.epoch % self.args.test_every == 0:
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
