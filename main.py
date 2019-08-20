# python main.py --lr=0.05 --lr_milestones 30 60 90 120 150 180 210 240 270 300 --lr_gamma=0.5 --wd=0.0005 --nesterov --momentum=0.9 --model="VGG('VGG11')" --epoch=300 --train_batch_size=128 --save_path="results/CIFAR-10/VGG-11/runs/run_1/metrics"

import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np

import argparse

from models import *
from misc import progress_bar, begin_chart, begin_per_epoch_chart, add_chart_point
from learn_utils import reset_seed


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='sgd momentum')
    parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
    parser.add_argument('--model', default="VGG('VGG19')", type=str, help='what model to use')
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

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=self.args.wd, nesterov=self.args.nesterov)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.lr_milestones, gamma=self.args.lr_gamma)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

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
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self,epoch,accuracy):
        model_out_path = "checkpoints/model_%s_%.2f%%.pth" % (epoch,accuracy * 100)
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()

        begin_per_epoch_chart("TrainAcc",self.args.save_path)
        begin_per_epoch_chart("TestAcc",self.args.save_path)
        begin_per_epoch_chart("TrainLoss",self.args.save_path)
        begin_per_epoch_chart("TestLoss",self.args.save_path)

        reset_seed(self.args.seed)
        accuracy = 0
        for epoch in range(1, self.args.epoch + 1):
            print("\n===> epoch: %d/%d" % (epoch,self.args.epoch))
            self.scheduler.step(epoch)

            train_result = self.train()
            add_chart_point("TrainAcc", epoch, train_result[1],self.args.save_path)
            add_chart_point("TrainLoss", epoch, train_result[0],self.args.save_path)

            test_result = self.test()
            add_chart_point("TestAcc", epoch, test_result[1],self.args.save_path)
            add_chart_point("TestLoss", epoch, test_result[0],self.args.save_path)

            if accuracy < test_result[1]:
                accuracy = test_result[1]
                self.save(epoch,accuracy)


if __name__ == '__main__':
    main()
