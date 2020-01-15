import sys
import argparse
import pickle
import os
from shutil import copyfile

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from tensorboardX import SummaryWriter
from torchvision import transforms as transforms

from learn_utils import reset_seed, EarlyStopping
from misc import progress_bar
from models import *

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    pass


CIFAR_10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
CIFAR_100_CLASSES = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
)


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--model', default="VGG('VGG19')",
                        type=str, help='what model to use')
    parser.add_argument('--dataset', default="CIFAR-10", type=str, choices=[
                        "CIFAR-10", "CIFAR-100"], help='What dataset to use. Options: CIFAR-10, CIFAR-100')
    parser.add_argument('--half', '-hf', action='store_true',help='use half precision')
    parser.add_argument('--mixpo',help='What Apex AMP opt level to use:\n 0 - FP32 training \n 1 - Conservative Mixed Precision, only some whitelist ops are done in FP16 \n 2 - Fast Mixed Precision, this is the standard mixed precision training. It maintains FP32 master weights and optimizer.step acts directly on the FP32 master weights \n 3 - FP16 training')
    parser.add_argument('--load_model', default="",type=str, help='what model to load')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.0,type=float, help='sgd momentum')
    parser.add_argument('--nesterov', action='store_true',help='Use nesterov momentum')
    parser.add_argument('--epoch', default=200, type=int,help='number of epochs tp train for')
    parser.add_argument('--train_batch_size', default=128,type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=512,type=int, help='testing batch size')
    parser.add_argument('--train_subset', default=None,type=int, help='Number of samples to train on')
    parser.add_argument('--initialization', '-init', default=0, type=int,
                        help='The type of initialization to be used \n 0 - Default pytorch initialization \n 1 - Xavier Initialization\n 2 - He et. al Initialization\n 3 - SELU Initialization\n 4 - Orthogonal Initialization')
    parser.add_argument('--initialization_batch_norm', '-init_batch',action='store_true', help='use batch norm initialization')

    parser.add_argument('--save_model', '-save',action='store_true', help='perform_top_down_sum')
    parser.add_argument('--save_interval', default=5,type=int, help='perform_top_down_sum')
    parser.add_argument('--save_dir', default="checkpoints",type=str, help='save dir name')

    parser.add_argument('--lr_milestones', nargs='+', type=int,
                        default=[30, 60, 90, 120, 150], help='Lr Milestones')
    parser.add_argument('--use_reduce_lr', action='store_true',
                        help='Use reduce lr on plateou')
    parser.add_argument('--reduce_lr_patience', type=int,
                        default=20, help='reduce lr patience')
    parser.add_argument('--reduce_lr_delta', type=float,
                        default=0.02, help='minimal difference to improve losss')
    parser.add_argument('--reduce_lr_min_lr', type=float,
                        default=0.0005, help='minimal lr')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='Lr gamma')
    parser.add_argument('--es_patience', type=int, default=10000, help='Early stopping pacience')

    parser.add_argument('--num_workers_train', default=4,
                        type=int, help='number of workers for loading train data')
    parser.add_argument('--num_workers_test', default=2,
                        type=int, help='number of workers for loading test data')

    parser.add_argument('--all_metrics', action='store_true',
                        help="In addition to accuracy and loss calculate all metrics available")
    parser.add_argument('--cuda', default=torch.cuda.is_available(),
                        type=bool, help='whether cuda is in use')
    parser.add_argument('--seed', default=None, type=int,
                        help='Seed to be used by randomizer')
    parser.add_argument('--progress_bar', '-pb',
                        action='store_true', help='Show the progress bar')
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
        self.es = EarlyStopping(patience=self.args.es_patience)
        if self.args.save_dir == "" or self.args.save_dir == None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir="runs/"+self.args.save_dir)
            with open("runs/"+self.args.save_dir+"/README.md", 'w+') as f:
                f.write(' '.join(sys.argv[1:]))
        self.batch_plot_idx = 0

        self.train_batch_plot_idx = 0
        self.test_batch_plot_idx = 0
        self.val_batch_plot_idx = 0
        if self.args.dataset == "CIFAR-10":
            self.nr_classes = len(CIFAR_10_CLASSES)
        elif self.args.dataset == "CIFAR-100":
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

        if self.args.dataset == "CIFAR-10":
            self.train_set = torchvision.datasets.CIFAR10(
                root='../storage', train=True, download=True, transform=train_transform)
        elif self.args.dataset == "CIFAR-100":
            self.train_set = torchvision.datasets.CIFAR100(
                root='../storage', train=True, download=True, transform=train_transform)

        if self.args.train_subset == None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_set, batch_size=self.args.train_batch_size, shuffle=True)
        else:
            filename = "subset_indices/subset_balanced_{}_{}.data".format(
                self.dataset, self.args.train_subset)
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    subset_indices = pickle.load(f)
            else:
                subset_indices = []
                per_class = self.args.train_subset//self.nr_classes
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
                dataset=self.train_set, batch_size=self.args.train_batch_size, sampler=SubsetRandomSampler(subset_indices))
            if self.args.validate:
                self.validate_loader = torch.utils.data.DataLoader(
                    dataset=self.train_set, batch_size=self.args.train_batch_size, sampler=SubsetRandomSampler(subset_indices))

        if self.args.dataset == "CIFAR-10":
            test_set = torchvision.datasets.CIFAR10(
                root='../storage', train=False, download=True, transform=test_transform)
        elif self.args.dataset == "CIFAR-100":
            test_set = torchvision.datasets.CIFAR100(
                root='../storage', train=False, download=True, transform=test_transform)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = eval(self.args.model)
        self.save_dir = "../storage/" + self.args.save_dir
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

        self.optimizer = optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=self.args.nesterov)
        if self.args.use_reduce_lr:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.args.lr_gamma, patience=self.args.reduce_lr_patience, min_lr=self.args.reduce_lr_min_lr, verbose=True, threshold=self.args.reduce_lr_delta)
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.args.lr_milestones, gamma=self.args.lr_gamma)
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        if self.cuda:
            if self.args.half:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

    def get_batch_plot_idx(self):
        self.batch_plot_idx += 1
        return self.batch_plot_idx - 1

    def train(self):
        print("train:")
        self.model.train()
        total_loss = 0
        correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
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
            total_loss += loss.item()
            self.writer.add_scalar("Train/Batch Loss", loss.item(), self.get_batch_plot_idx())

            prediction = torch.max(output, 1)
            total += target.size(0)

            correct += torch.sum((prediction[1] == target).float())

            pred_labels = torch.nn.functional.one_hot(prediction[1], num_classes=10)
            true_labels = torch.nn.functional.one_hot(target, num_classes=10)

            TP += torch.sum((pred_labels == 1) & (true_labels == 1))
            TN += torch.sum((pred_labels == 0) & (true_labels == 0))
            FP += torch.sum((pred_labels == 1) & (true_labels == 0))
            FN += torch.sum((pred_labels == 0) & (true_labels == 1))

            if self.args.progress_bar:
                progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (total_loss / (batch_num + 1), 100.0 * correct/total, correct, total))

        return total_loss, correct / total, TP, TN, FP, FN

    def test(self):
        print("test:")
        self.model.eval()
        total_loss = 0
        correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.writer.add_scalar("Test/Batch Loss", loss.item(), self.get_batch_plot_idx())
                total_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)

                correct += torch.sum((prediction[1] == target).float())

                pred_labels = torch.nn.functional.one_hot(prediction[1], num_classes=10)
                true_labels = torch.nn.functional.one_hot(target, num_classes=10)

                TP += torch.sum((pred_labels == 1) & (true_labels == 1))
                TN += torch.sum((pred_labels == 0) & (true_labels == 0))
                FP += torch.sum((pred_labels == 1) & (true_labels == 0))
                FN += torch.sum((pred_labels == 0) & (true_labels == 1))

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                 % (total_loss / (batch_num + 1), 100. * correct / total, correct, total))

        return total_loss, correct/total, TP, TN, FP, FN

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
        if not self.args.seed is None: 
            reset_seed(self.args.seed)
        self.load_data()
        self.load_model()

        best_accuracy = 0
        try:
            for epoch in range(1, self.args.epoch + 1):
                print("\n===> epoch: %d/%d" % (epoch, self.args.epoch))

                train_result = self.train()

                # Took the metrics from here: https://en.wikipedia.org/wiki/Precision_and_recall
                loss = train_result[0]
                accuracy = train_result[1]
                self.writer.add_scalar("Train/Loss", loss, epoch)
                self.writer.add_scalar("Train/Accuracy", accuracy, epoch)
                if self.args.all_metrics:
                    TP = train_result[2]
                    TN = train_result[3]
                    FP = train_result[4]
                    FN = train_result[5]
                    TPR = TP/(TP+FN)
                    TNR = TN/(TN+FP)
                    PPV = TP/(TP+FP)
                    NPV = TN/(TN+FN)
                    FNR = FN/(FN+TP)
                    FPR = FP/(FP+TN)
                    FDR = FP/(FP+TP)
                    FOR = FN/(FN+TN)
                    TS = TP/(TP+FN+FP)
                    F1 = (2*TP)/(2*TP+FP+FN)
                    MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                    BM = TPR+TNR-1
                    MK = PPV+NPV-1

                    self.writer.add_scalar("Train/F1 score", F1, epoch)
                    self.writer.add_scalar("Train/Sensitivity", TPR, epoch)
                    self.writer.add_scalar("Train/Specificity", TNR, epoch)
                    self.writer.add_scalar("Train/Precision", PPV, epoch)
                    self.writer.add_scalar("Train/Negative predictive value", NPV, epoch)
                    self.writer.add_scalar("Train/Miss rate", FNR, epoch)
                    self.writer.add_scalar("Train/Fall-out", FPR, epoch)
                    self.writer.add_scalar("Train/False discovery rate ", FDR, epoch)
                    self.writer.add_scalar("Train/False omission rate ", FOR, epoch)
                    self.writer.add_scalar("Train/Threat score", TS, epoch)
                    self.writer.add_scalar("Train/TP", TP, epoch)
                    self.writer.add_scalar("Train/TN", TN, epoch)
                    self.writer.add_scalar("Train/FP", FP, epoch)
                    self.writer.add_scalar("Train/FN", FN, epoch)
                    self.writer.add_scalar("Train/Matthews correlation coefficient", MCC, epoch)
                    self.writer.add_scalar("Train/Informedness", BM, epoch)
                    self.writer.add_scalar("Train/Markedness", MK, epoch)

                test_result = self.test()

                loss = test_result[0]
                accuracy = test_result[1]
                self.writer.add_scalar("Test/Loss", loss, epoch)
                self.writer.add_scalar("Test/Accuracy", accuracy, epoch)
                if self.args.all_metrics:
                    TP = test_result[2]
                    TN = test_result[3]
                    FP = test_result[4]
                    FN = test_result[5]
                    TPR = TP/(TP+FN)
                    TNR = TN/(TN+FP)
                    PPV = TP/(TP+FP)
                    NPV = TN/(TN+FN)
                    FNR = FN/(FN+TP)
                    FPR = FP/(FP+TN)
                    FDR = FP/(FP+TP)
                    FOR = FN/(FN+TN)
                    TS = TP/(TP+FN+FP)
                    F1 = (2*TP)/(2*TP+FP+FN)
                    MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                    BM = TPR+TNR-1
                    MK = PPV+NPV-1

                    self.writer.add_scalar("Test/F1 score", F1, epoch)
                    self.writer.add_scalar("Test/Sensitivity", TPR, epoch)
                    self.writer.add_scalar("Test/Specificity", TNR, epoch)
                    self.writer.add_scalar("Test/Precision", PPV, epoch)
                    self.writer.add_scalar("Test/Negative predictive value", NPV, epoch)
                    self.writer.add_scalar("Test/Miss rate", FNR, epoch)
                    self.writer.add_scalar("Test/Fall-out", FPR, epoch)
                    self.writer.add_scalar("Test/False discovery rate ", FDR, epoch)
                    self.writer.add_scalar("Test/False omission rate ", FOR, epoch)
                    self.writer.add_scalar("Test/Threat score", TS, epoch)
                    self.writer.add_scalar("Test/TP", TP, epoch)
                    self.writer.add_scalar("Test/TN", TN, epoch)
                    self.writer.add_scalar("Test/FP", FP, epoch)
                    self.writer.add_scalar("Test/FN", FN, epoch)
                    self.writer.add_scalar("Test/Matthews correlation coefficient", MCC, epoch)
                    self.writer.add_scalar("Test/Informedness", BM, epoch)
                    self.writer.add_scalar("Test/Markedness", MK, epoch)

                self.writer.add_scalar("Model/Norm", self.get_model_norm(), epoch)
                self.writer.add_scalar("Train Params/Learning rate", self.scheduler.get_last_lr()[0], epoch)

                if best_accuracy < test_result[1]:
                    best_accuracy = test_result[1]
                    self.save(epoch, best_accuracy)
                    print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))

                if self.args.save_model and epoch % self.args.save_interval == 0:
                    self.save(epoch, 0)

                if self.args.use_reduce_lr:
                    self.scheduler.step(train_result[0])
                else:
                    self.scheduler.step()

                if self.es.step(train_result[0]):
                        raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
        
        print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))
        files = os.listdir(self.save_dir)
        paths = [os.path.join(self.save_dir, basename) for basename in files if "_0_" not in basename]
        if len(paths) > 0:
            src = max(paths, key=os.path.getctime)
            copyfile(src, os.path.join("runs",self.args.save_dir,os.path.basename(src)))
            
        with open("runs/"+self.args.save_dir+"/README.md", 'a+') as f:
            f.write("\n## Accuracy\n %.3f%%" % (best_accuracy * 100))
        print("Saved best accuracy checkpoint")


    def get_model_norm(self, norm_type=2):
        norm = 0.0
        for param in self.model.parameters():
            norm += torch.norm(input=param, p=norm_type, dtype=torch.float)
        return norm


if __name__ == '__main__':
    main()
