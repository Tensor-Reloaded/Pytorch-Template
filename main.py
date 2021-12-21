import collections
import sys
import pprint
import argparse
import pickle
import os
import re
from multiprocessing import Process, freeze_support
from shutil import copyfile
import pandas as pd

import numpy as np
from collections import OrderedDict
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
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import cohen_kappa_score

from utils import *
from utils.misc import progress_bar, save_current_code
import models



@hydra.main(config_path='configs', config_name='config')
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
    save_current_code(save_config_path)

    solver = Solver(config)
    return solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.args = config
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = self.args.device
        self.cuda = True if 'cuda' in self.device else False
        self.train_loader = None
        self.val_loader = None
        self.infer_loader = None
        self.es = EarlyStopping(patience=self.args.es_patience)
        self.scaler = GradScaler(enabled=self.args.half and self.args.grad_scaler)


        if not self.args.save_dir:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir="runs/" + self.args.save_dir)

        self.train_batch_plot_idx = 0
        self.val_batch_plot_idx = 0

    def construct_transformations(self, transformation_config):
        transformations_config = OmegaConf.load(to_absolute_path(f'configs/transformations/{transformation_config}.yaml'))

        cache_index = None
        transformation_list = []
        for idx, (name,parameters) in enumerate(transformations_config.items()):
            if name not in transformations:
                print(f"This transformation is not implemented ({name}), go ahead and commit it")
                exit()

            apply_to = None
            if 'apply_to' in parameters:
                apply_to = parameters.pop('apply_to')
            transformation = transformations[name]
            if cache_index is None and not transformation['cacheable']:
                cache_index = idx
            transformation_fnc = TransformWrapper(transformation['constructor'](**parameters), apply_to)
            transformation_list.append(transformation_fnc)
        if cache_index is None:
            cache_index = idx
        transformation_list = transforms.Compose(transformation_list) if len(transformation_list) > 0 else None
        return transformation_list, cache_index

    def construct_dataset(self, dataset_config, transformations, cache_index=0):
        parameters = OmegaConf.to_container(dataset_config.load_params, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}

        dataset = MemoryStoredDataset(dataset=datasets[dataset_config.name](**parameters), transformations=transformations, save_in_memory=dataset_config.save_in_memory, cache_index=cache_index)

        return dataset

    def construct_dataloader(self, dataset_config, dataset):
        if hasattr(dataset_config, 'mixup_args') and dataset_config.mixup_args != None:
            collate_fn = FastCollateMixup(**dataset_config.mixup_args)
        else:
            collate_fn = None

        if not hasattr(dataset_config, 'subset') or dataset_config.subset is None or dataset_config.subset == '' or dataset_config.subset <= 0:
            sampler = None
        else:
            indices = ((np.random.random(len(dataset)) < dataset_config.subset).nonzero()[0]).tolist()
            sampler = SubsetRandomSampler(indices)
            dataset_config.shuffle = False
        
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=dataset_config.batch_size, shuffle=(dataset_config.shuffle and sampler is None), sampler=sampler, num_workers=dataset_config.num_workers, collate_fn=collate_fn, pin_memory=dataset_config.pin_memory, drop_last=dataset_config.drop_last, persistent_workers=dataset_config.num_workers>0)
        return loader

    def init_dataset(self):
        if hasattr(self.args, 'train_dataset'):
            if self.args.train_dataset.name not in datasets:
                print(f"This dataset is not implemented ({self.args.train_dataset.name}), go ahead and commit it")
                exit()
            if hasattr(self.args.train_dataset, 'transform'):
                train_transformations, train_cache_index = self.construct_transformations(self.args.train_dataset.transform)
            else:
                train_transformations, train_cache_index = None, None
            self.train_set = self.construct_dataset(self.args.train_dataset, train_transformations, train_cache_index)
            self.train_loader = self.construct_dataloader(self.args.train_dataset, self.train_set)

    
        if hasattr(self.args, 'val_dataset'):
            if self.args.val_dataset.name not in datasets:
                print(f"This dataset is not implemented ({self.args.val_dataset.name}), go ahead and commit it")
                exit()
            if hasattr(self.args.val_dataset, 'transform'):
                val_transformations, val_cache_index = self.construct_transformations(self.args.val_dataset.transform)
            else:
                val_transformations, val_cache_index = None, None
            self.val_set = self.construct_dataset(self.args.val_dataset, val_transformations, val_cache_index)
            self.val_loader = self.construct_dataloader(self.args.val_dataset, self.val_set)

        if hasattr(self.args, 'infer_dataset'):
            if self.args.infer_dataset.name not in datasets:
                print(f"This dataset is not implemented ({self.args.infer_dataset.name}), go ahead and commit it")
                exit()
            if hasattr(self.args.infer_dataset, 'transform'):
                infer_transformations, infer_cache_index = self.construct_transformations(self.args.infer_dataset.transform)
            else:
                infer_transformations, infer_cache_index = None, None
            self.infer_set = self.construct_dataset(self.args.infer_dataset, infer_transformations, infer_cache_index)
            self.infer_loader = self.construct_dataloader(self.args.infer_dataset, self.infer_set)

        self.output_transformations = None
        if hasattr(self.args, 'output_transformation'):
            self.output_transformations, _ = self.construct_transformations(self.args.output_transformation)



    def init_model(self):
        if self.cuda:
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
                    nn.init.kaiming_normal_(m.weight, mode='fan_in')
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
            if self.args.model.name == "Recorder":
                loaded_model = torch.load(self.args.load_model, map_location=self.device)
                new_state_dict = OrderedDict()

                for key, value in loaded_model.items():
                    # if key.startswith('net.transformer.'):
                        # new_state_dict[key[16:]] = value
                    if key.startswith('net.'):
                        new_state_dict[key[4:]] = value

                # self.model.vit.transformer.load_state_dict(new_state_dict)
                self.model.vit.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(torch.load(self.args.load_model, map_location=self.device))
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

        
        if self.args.optimizer.use_SAM:
            self.optimizer = optimizers['SAM'](params=parameters["params"],base_optimizer=self.optimizer,rho=self.args.optimizer.SAM_rho)
        else:
            self.optimizer = self.optimizer(**parameters)
        
        if self.args.optimizer.use_lookahead:
            self.optimizer = torch_optimizer.Lookahead(self.optimizer, k=self.args.optimizer.lookahead_k, alpha=self.args.optimizer.lookahead_alpha)

    def init_scheduler(self):
        (name,parameters) = self.args.scheduler.items()[0]
        self.scheduler_name = name
        if name not in schedulers:
            print(f"This scheduler is not implemented ({name}), go ahead and commit it")
            exit()

        parameters = OmegaConf.to_container(parameters, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        parameters["optimizer"] = self.optimizer
        self.scheduler = schedulers[name](**parameters)

    def init_criterion(self):
        (name,parameters) = self.args.loss.items()[0]
        if name not in losses:
            print(f"This loss is not implemented ({name}), go ahead and commit it")
            exit()

        parameters = OmegaConf.to_container(parameters, resolve=True)
        parameters = {k: v for k, v in parameters.items() if v is not None}
        parameters["device"] = self.device

        self.criterion = losses[name]['constructor'](**parameters)

    def init_metrics(self):
        self.metrics = {
            'train':{
                'batch':[],
                'epoch':[]
            },
            'val':{
                'batch':[],
                'epoch':[]
            },
            'solver':{
                'batch':[],
                'epoch':[]
            },
        }


        for (name,args) in self.args.train_metrics.items():
            if name not in metrics:
                print(f"This metric is not implemented ({name}), go ahead and commit it")
                exit()

            if not hasattr(args, 'index'):
                metric_index = None
            else:
                metric_index = args.index

            if args.parameters is None:
                args.parameters = {}
            metric_func = metrics[name]['constructor'](**args.parameters)
            metric_object = Metric(name, metric_func, index=metric_index, solver_metric=False, aggregator=args.aggregator)
            for level in args.levels:
                self.metrics['train'][level].append(metric_object)

        for (name,args)  in self.args.val_metrics.items():
            if name not in metrics:
                print(f"This metric is not implemented ({name}), go ahead and commit it")
                exit()

            if not hasattr(args, 'index'):
                metric_index = None
            else:
                metric_index = args.index
                
            if args.parameters is None:
                args.parameters = {}
            metric_func = metrics[name]['constructor'](**args.parameters)
            metric_object = Metric(name, metric_func, index=metric_index, solver_metric=False, aggregator=args.aggregator)
            for level in args.levels:
                self.metrics['val'][level].append(metric_object)

        for (name,args)  in self.args.solver_metrics.items():
            if name not in metrics:
                print(f"This metric is not implemented ({name}), go ahead and commit it")
                exit()

            if args.parameters is None:
                args.parameters = {}
            metric_func = metrics[name]['constructor'](**args.parameters)
            metric_object = Metric(name, metric_func, index=None, solver_metric=True, aggregator=args.aggregator)
            for level in args.levels:
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

        # accumulation_data = []
        # accumulation_target = []

        predictions = []
        targets = []
        for batch_num, (data, target) in enumerate(self.train_loader):
            if isinstance(data,list) or isinstance(data,tuple):
                data = [i.to(self.device) for i in data]
            else:
                data = data.to(self.device)
            if isinstance(target,list) or isinstance(target,tuple):
                target = [i.to(self.device) for i in target]
            else:
                target = target.to(self.device)

            # if self.args.optimizer.use_SAM:
            #     accumulation_data.append(data)
            #     accumulation_target.append(target)


            
            def sam_closure():
                self.disable_bn()
                while True: 
                    with autocast(enabled=self.args.half):
                        output = self.model(data)
                        if self.output_transformations is not None:
                            output = self.output_transformations(output)
                        
                        if hasattr(self.args.model, 'returns_loss') and self.args.model.returns_loss:
                            loss = output
                        else:
                            loss = self.criterion(output, target)
                        loss = loss / self.args.train_dataset.update_every

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

                    if self.args.optimizer.batch_replay:
                        found_inf = False
                        for _, param in self.model.named_parameters():
                            if  param.grad.isnan().any() or param.grad.isinf().any():
                                found_inf = True
                                break
                        if found_inf:
                            self.scaler.update()
                            self.optimizer.zero_grad()#(set_to_none=True)
                            if type(self.args.optimizer.batch_replay) == int or type(self.args.optimizer.batch_replay) == float: 
                                self.args.optimizer.batch_replay -= 1
                        else:
                            break
                    else:
                        break

                if self.train_batch_plot_idx % self.args.train_dataset.update_every == 0:
                    self.scaler.unscale_(self.optimizer)
                    
                    if self.args.optimizer.max_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optimizer.max_norm)
                    self.enable_bn()
                



            while True: 
                with autocast(enabled=self.args.half):
                    output = self.model(data)
                    if self.output_transformations is not None:
                        output = self.output_transformations(output)
                    
                    if hasattr(self.args.model, 'returns_loss') and self.args.model.returns_loss:
                        loss = output
                    else:
                        loss = self.criterion(output, target)
                    loss = loss / self.args.train_dataset.update_every

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

                if self.args.optimizer.batch_replay:
                    found_inf = False
                    for _, param in self.model.named_parameters():
                        if  param.grad.isnan().any() or param.grad.isinf().any():
                            found_inf = True
                            break
                    if found_inf:
                        self.scaler.update()
                        self.optimizer.zero_grad()#(set_to_none=True)
                        if type(self.args.optimizer.batch_replay) == int or type(self.args.optimizer.batch_replay) == float: 
                            self.args.optimizer.batch_replay -= 1
                    else:
                        break
                else:
                    break

            if self.train_batch_plot_idx % self.args.train_dataset.update_every == 0:
                self.scaler.unscale_(self.optimizer)
                
                if self.args.optimizer.max_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optimizer.max_norm)
                
                if self.args.optimizer.use_SAM:
                    self.scaler.step(self.optimizer, closure=sam_closure)
                    # self.model.update_moving_average()
                else:
                    self.scaler.step(self.optimizer)
                    # self.model.update_moving_average()

                self.scaler.update()

                self.optimizer.zero_grad()#(set_to_none=True)
                
                if self.scheduler_name == "OneCycleLR":
                    self.scheduler.step()


            if isinstance(output,list) or isinstance(output,tuple):
                for pred_idx, o in enumerate(output):
                    o=o.cpu()
                    if len(predictions) <= pred_idx:
                        predictions.append(torch.tensor(o))
                    else:
                        predictions[pred_idx] = torch.cat((predictions[pred_idx], o))
            else:
                output=output.cpu()
                if isinstance(output, torch.Tensor):
                    predictions = output.clone().detach()
                else:
                    predictions = torch.tensor(output)

            if isinstance(target,list) or isinstance(target,tuple):
                for target_idx, t in enumerate(target):
                    t=t.cpu()
                    if len(targets) <= target_idx:
                        targets.append(torch.tensor(t))
                    else:
                        targets[target_idx] = torch.cat((targets[target_idx], t))
            else:
                target=target.cpu()
                if isinstance(target, torch.Tensor):
                    targets = target.clone().detach()
                else:
                    targets = torch.tensor(target)
            
            metrics_results = {}
            for metric in self.metrics['train']['batch']:
                metrics_results["Train/Batch-"+metric.name] = metric.calculate(output, target, level='batch')

            for metric in self.metrics['solver']['batch']:
                    metrics_results["Solver/Batch-"+metric.name] = metric.calculate(solver = self, level='batch')

            print_metrics(self.writer, metrics_results, self.get_train_batch_plot_idx())

            if self.args.progress_bar:
                progress_bar(batch_num, len(self.train_loader))

        return predictions, targets

    def val(self):
        print("val:")
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        predictions = []
        targets = []
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.val_loader):
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
                    if self.output_transformations is not None:
                        output = self.output_transformations(output)
                        
                    if hasattr(self.args.model, 'returns_loss') and self.args.model.returns_loss:
                        loss = output
                    else:
                        loss = self.criterion(output, target)


                if isinstance(output,list) or isinstance(output,tuple):
                    for pred_idx, o in enumerate(output):
                        o=o.cpu()
                        if len(predictions) <= pred_idx:
                            predictions.append(torch.tensor(o))
                        else:
                            predictions[pred_idx] = torch.cat((predictions[pred_idx], o))
                else:
                    output=output.cpu()
                    if isinstance(output, torch.Tensor):
                        predictions = output
                    else:
                        predictions = torch.tensor(output)

                if isinstance(target,list) or isinstance(target,tuple):
                    for target_idx, t in enumerate(target):
                        t=t.cpu()
                        if len(targets) <= target_idx:
                            targets.append(torch.tensor(t))
                        else:
                            targets[target_idx] = torch.cat((targets[target_idx], t))
                else:
                    target=target.cpu()
                    if isinstance(target, torch.Tensor):
                        targets = target
                    else:
                        targets = torch.tensor(target)

                metrics_results = {}
                for metric in self.metrics['val']['batch']:
                    metrics_results["Val/Batch-"+metric.name] = metric.calculate(output, target, level='batch')


                print_metrics(self.writer, metrics_results, self.get_val_batch_plot_idx())

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.val_loader))

        return predictions, targets

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
                    if self.output_transformations is not None:
                        output = self.output_transformations(output)

                
                if isinstance(output,list) or isinstance(output,tuple):
                    for pred_idx, o in enumerate(output):
                        o=o.cpu()
                        if len(predictions) <= pred_idx:
                            predictions.append(torch.tensor(o))
                        else:
                            predictions[pred_idx] = torch.cat((predictions[pred_idx], o))
                else:
                    output=output.cpu()
                    if isinstance(output, torch.Tensor):
                        predictions = output
                    else:
                        predictions = torch.tensor(output)
        
                
                filenames.extend(filename)

        return filenames, predictions


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
        self.init_dataset()
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
                    result = metric.calculate(predictions, targets, level='epoch')
                    if type(result) is dict:
                        for each_key in result.keys():
                            metrics_results[metric_name+"_{0}".format(each_key)] = result[each_key] 
                    else:
                        metrics_results[metric_name] = result

                if self.epoch % self.args.val_every == 0:
                    predictions, targets = self.val()
                    for metric in self.metrics['val']['epoch']:
                        metric_name = "Val/"+metric.name
                        result = metric.calculate(predictions, targets, level='epoch')
                        if type(result) is dict:
                            for each_key in result.keys():
                                metrics_results[metric_name+"_{0}".format(each_key)] = result[each_key] 
                        else:
                            metrics_results[metric_name] = result

                for metric in self.metrics['solver']['epoch']:
                    metric_name = "Solver/"+metric.name
                    metrics_results[metric_name] = metric.calculate(solver = self, level='epoch')

                print_metrics(self.writer, metrics_results, self.epoch)

                
                if self.epoch % self.args.val_every == 0:
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

                if self.scheduler_name == "MultiStepLR":
                    self.scheduler.step()
                elif self.scheduler_name == "ReduceLROnPlateau":
                    self.scheduler.step(metrics_results[self.args.scheduler_metric])
                elif self.scheduler_name == "OneCycleLR":
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

    def get_val_batch_plot_idx(self):
        self.val_batch_plot_idx += 1
        return self.val_batch_plot_idx - 1



if __name__ == '__main__':
    main()
