# python main.py --lr=0.05 --lr_milestones 30 60 90 120 150 180 210 240 270 300 --lr_gamma=0.5 --wd=0.0005 --nesterov --momentum=0.9 --model="VGG('VGG11')" --epoch=300 --train_batch_size=128 --save_path="results/CIFAR-10/VGG-11/runs/run_1/metrics"
import os
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
from torchvision import transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import argparse

from models import *
from misc import progress_bar
from learn_utils import reset_seed


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='sgd momentum')
    parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
    parser.add_argument('--model', default="VGG('VGG19')", type=str, help='what model to use')
    parser.add_argument('--half', '-hf', action='store_true', help='use half precision')
    parser.add_argument('--initialization', '-init', default=0, type=int, help='The type of initialization to be used \n 0 - Default pytorch initialization \n 1 - Xavier Initialization\n 2 - He et. al Initialization\n 3 - SELU Initialization\n 4 - Orthogonal Initialization')
    parser.add_argument('--initialization_batch_norm', '-init_batch', action='store_true', help='use batch norm initialization')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--train_batch_size', default=128, type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=512, type=int, help='testing batch size')
    parser.add_argument('--num_workers_train', default=4, type=int, help='number of workers for loading train data')
    parser.add_argument('--num_workers_test', default=2, type=int, help='number of workers for loading test data')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--nesterov', action='store_true', help='Use nesterov momentum')
    parser.add_argument('--save_path', default="results", type=str, help='path to folder where results should be saved')
    parser.add_argument('--seed', default=0, type=int, help='Seed to be used by randomizer')
    parser.add_argument('--lr_milestones', nargs='+', type=int,default=[30, 60, 90, 120, 150], help='Lr Milestones')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='Lr gamma')
    parser.add_argument('--progress_bar', '-pb', action='store_true', help='Show the progress bar')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


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
        self.writer = SummaryWriter()
        self.batch_plot_idx = 0


    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='../storage', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.args.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='../storage', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = eval(self.args.model).to(self.device)
        
        #TODO Fix error for: python main.py --lr=0.05 --lr_milestones 30 60 90 120 150 180 210 240 270 300 --lr_gamma=0.5 --wd=0.0005 --nesterov --momentum=0.9 --model="VGG('VGG11')" --epoch=300 --train_batch_size=128 --half
        if self.cuda:
            if self.args.half:
                self.model.half()
                for layer in self.model.modules():
                    if isinstance(layer, nn.BatchNorm2d):
                        layer.float()
                print("Using half precision")

        if self.args.initialization == 1:
            #xavier init
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
        elif self.args.initialization == 2:
            # he initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal(m.weight, mode='fan_in')
        elif self.args.initialization == 3:
            # selu init
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    nn.init.normal(m.weight, 0, sqrt(1. / fan_in))
                elif isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    nn.init.normal(m.weight, 0, sqrt(1. / fan_in))
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

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=self.args.wd, nesterov=self.args.nesterov)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.lr_milestones, gamma=self.args.lr_gamma)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def get_batch_plot_idx(self):
        self.batch_plot_idx += 1
        return self.batch_plot_idx - 1

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            if self.device == torch.device('cuda') and self.args.half:
                # data, target = data.half(), target.half()
                data = data.half()
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            self.writer.add_scalar("Loss/train batch", loss.item(), self.get_batch_plot_idx())
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            if self.args.progress_bar:
                progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                    % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                if self.device == 'cuda' and self.args.half:
                    data = data.half()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.writer.add_scalar("Loss/test batch", loss.item(), self.get_batch_plot_idx())
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                        % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self,epoch,accuracy):
        os.makedirs('checkpoints', exist_ok=True)
        model_out_path = "checkpoints/model_%s_%.2f%%.pth" % (epoch,accuracy * 100)
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()

        reset_seed(self.args.seed)
        accuracy = 0
        for epoch in range(1, self.args.epoch + 1):
            print("\n===> epoch: %d/%d" % (epoch,self.args.epoch))
            self.scheduler.step(epoch)

            train_result = self.train()

            self.writer.add_scalar("Loss/train", train_result[0], epoch)
            self.writer.add_scalar("Acc/train", train_result[1], epoch)

            test_result = self.test()

            self.writer.add_scalar("Loss/test", test_result[0], epoch)
            self.writer.add_scalar("Acc/test", test_result[1], epoch)

            self.writer.add_scalar("Model/norm", self.get_model_norm(), epoch)
            self.writer.add_scalar("Train Params/lr", self.scheduler.get_lr()[0], epoch)

            if accuracy < test_result[1]:
                accuracy = test_result[1]
                self.save(epoch,accuracy)

    def get_model_norm(self, norm_type = 2):
        norm = 0
        for param in self.model.parameters():
            norm += torch.norm(param, p=norm_type)
        return norm

if __name__ == '__main__':
    main()
