import collections
import sys
import pprint
import argparse
import pickle
import os
import re
from shutil import copyfile

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch_optimizer
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as transforms
import hydra
from hydra import utils
from omegaconf import DictConfig, OmegaConf

from utils import *
from misc import progress_bar
from models import *

APEX_MISSING = False
try:
    from apex import amp, optimizers
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("Apex not found on the system, it won't be using half-precision")
    APEX_MISSING = True
    pass


storage_dir = "../storage/"

@hydra.main(config_path='experiments', config_name='config')
def main(config: DictConfig):
    global storage_dir

    head, tail = os.path.split(config.save_dir)
    if tail == 'None':
        config.save_dir = head
    else:
        config.save_dir = os.path.join(head, tail)

    storage_dir = os.path.dirname(utils.get_original_cwd()) + "/storage/"
    save_config_path = "runs/" + config.save_dir
    os.makedirs(save_config_path, exist_ok=True)
    with open(os.path.join(save_config_path, "README.md"), 'w+') as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))

    if APEX_MISSING:
        config.half = False

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
        self.es = EarlyStopping(patience=self.args.optimizer.es_patience)
        
        if not self.args.save_dir:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir="runs/" + self.args.save_dir)

        self.train_batch_plot_idx = 0
        self.test_batch_plot_idx = 0
        if self.args.dataset.name == "CIFAR-10":
            self.nr_classes = len(CIFAR_10_CLASSES)
        elif self.args.dataset.name == "CIFAR-100":
            self.nr_classes = len(CIFAR_100_CLASSES)

    def load_data(self):
        if "CIFAR" in self.args.dataset:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(
            ), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), normalize])
        else:
            train_transform = transforms.Compose([transforms.ToTensor()])
            test_transform = transforms.Compose([transforms.ToTensor()])

        if self.args.dataset.name == "CIFAR-10":
            self.train_set = torchvision.datasets.CIFAR10(
                root=storage_dir, train=True, download=True, transform=train_transform)
        elif self.args.dataset.name == "CIFAR-100":
            self.train_set = torchvision.datasets.CIFAR100(
                root=storage_dir, train=True, download=True, transform=train_transform)

        if self.args.dataset.train_subset is None:
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.args.dataset.train_batch_size, shuffle=self.args.dataset.shuffle, num_workers=self.args.dataset.num_workers_train)
        else:
            filename = "subset_indices/subset_balanced_{}_{}.data".format(
                self.args.dataset, self.args.train_subset)
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    subset_indices = pickle.load(f)
            else:
                subset_indices = []
                per_class = self.args.train_subset // self.nr_classes
                targets = torch.tensor(self.train_set.targets)
                for i in range(self.nr_classes):
                    idx = (targets == i).nonzero().view(-1)
                    perm = torch.randperm(idx.size(0))[:per_class]
                    subset_indices += idx[perm].tolist()
                if not os.path.isdir("subset_indices"):
                    os.makedirs("subset_indices")
                with open(filename, 'wb') as f:
                    pickle.dump(subset_indices, f)
            subset_indices = torch.LongTensor(subset_indices)
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_set, batch_size=self.args.train_batch_size,
                sampler=SubsetRandomSampler(subset_indices),
                num_workers=self.args.dataset.num_workers_train)
            if self.args.validate:
                self.validate_loader = torch.utils.data.DataLoader(
                    dataset=self.train_set, batch_size=self.args.train_batch_size,
                    sampler=SubsetRandomSampler(subset_indices),
                    num_workers=self.args.dataset.num_workers_test)

        if self.args.dataset.name == "CIFAR-10":
            self.test_set = torchvision.datasets.CIFAR10(
                root=storage_dir, train=False, download=True, transform=test_transform)
        elif self.args.dataset.name == "CIFAR-100":
            self.test_set = torchvision.datasets.CIFAR100(
                root=storage_dir, train=False, download=True, transform=test_transform)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set, batch_size=self.args.dataset.test_batch_size, shuffle=False, num_workers=self.args.dataset.num_workers_test)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda' + ":" + str(self.args.cuda_device))
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = eval(self.args.model_command)
        self.save_dir = storage_dir + self.args.save_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.init_model()
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
        self.criterion = losses[self.args.loss.name](**parameters)
        

    def get_train_batch_plot_idx(self):
        self.train_batch_plot_idx += 1
        return self.train_batch_plot_idx - 1

    def get_test_batch_plot_idx(self):
        self.test_batch_plot_idx += 1
        return self.test_batch_plot_idx - 1

    def train(self):
        print("train:")
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            if self.args.half:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            print_batch_metrics(True, self.writer,{
                'loss':batch_loss
            }, self.get_train_batch_plot_idx())

            prediction = torch.max(output, 1)
            total += target.size(0)

            correct += torch.sum((prediction[1] == target).float()).item()

            if self.args.progress_bar:
                progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (total_loss / (batch_num + 1), 100.0 * correct/total, correct, total))
            if self.args.scheduler.name == "OneCycleLR":
                self.scheduler.step()
        return total_loss, correct / total

    def test(self):
        print("test:")
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                batch_loss = loss.item()
                total_loss += batch_loss
                print_batch_metrics(False, self.writer,{
                    'loss':batch_loss
                }, self.get_test_batch_plot_idx())

                prediction = torch.max(output, 1)
                total += target.size(0)

                correct += torch.sum((prediction[1] == target).float()).item()

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                 % (total_loss / (batch_num + 1), 100. * correct / total, correct, total))

        return total_loss, correct/total

    def save(self, epoch, accuracy, tag=None):
        if tag != None:
            tag = "_"+tag
        else:
            tag = ""
        model_out_path = self.save_dir + \
            "/model_{}_{}{}.pth".format(
                epoch, accuracy * 100, tag)
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        if self.args.seed is not None:
            reset_seed(self.args.seed)
        self.load_data()
        self.load_model()
        self.init_optimizer()
        self.init_scheduler()
        self.init_criterion()

        if self.cuda:
            if self.args.half:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=f"O{self.args.mixpo}",
                                                            patch_torch_functions=True, keep_batchnorm_fp32=True)
        best_accuracy = 0
        best_loss = np.inf
        try:
            for epoch in range(1, self.args.optimizer.epochs + 1):
                print("\n===> epoch: %d/%d" % (epoch, self.args.optimizer.epochs))
                self.epoch = epoch

                train_result = self.train()
                print_epoch_metrics(True, self.writer,{
                    'loss':train_result[0],
                    'accuracy':train_result[1]
                }, self.epoch)

                test_result = self.test()
                print_epoch_metrics(False, self.writer,{
                    'loss':test_result[0],
                    'accuracy':test_result[1],
                    'lr': self.scheduler.get_last_lr()[0],
                    'norm': self.get_model_norm()
                }, self.epoch)
                
                if best_loss > test_result[0]:
                    best_loss = test_result[0]

                if best_accuracy < test_result[1]:
                    best_accuracy = test_result[1]
                    self.save(epoch, best_accuracy)
                    print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))

                if self.args.save_model and epoch % self.args.save_interval == 0:
                    self.save(epoch, 0)

                if self.args.scheduler.name == "MultiStepLR":
                    self.scheduler.step()
                elif self.args.scheduler.name == "ReduceLROnPlateau":
                    self.scheduler.step(train_result[0])
                elif self.args.scheduler.name == "OneCycleLR":
                    pass
                else:
                    self.scheduler.step()

                if self.es.step(train_result[0]):
                    print("Early stopping")
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass

        print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))
        files = os.listdir(self.save_dir)
        paths = [os.path.join(self.save_dir, basename) for basename in files if "_0" not in basename]
        if len(paths) > 0:
            src = max(paths, key=os.path.getctime)
            copyfile(src, os.path.join("runs", self.args.save_dir, os.path.basename(src)))

        with open("runs/" + self.args.save_dir + "/README.md", 'a+') as f:
            f.write("\n## Accuracy\n %.3f%%" % (best_accuracy * 100))
        print("Saved best accuracy checkpoint")

        if self.args.optimized_metric == "accuracy":
            return 1-best_accuracy

        if self.args.optimized_metric == "loss":
            return best_loss


    def get_model_norm(self, norm_type=2):
        norm = 0.0
        for param in self.model.parameters():
            norm += torch.norm(input=param, p=norm_type, dtype=torch.float)
        return norm

    def init_model(self):
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


if __name__ == '__main__':
    main()
