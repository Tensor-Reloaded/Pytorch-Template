import os

import nibabel as nb
from skimage import io
from skimage import transform
from skimage.morphology import binary_erosion
from skimage.transform import resize, rotate

import pandas as pd
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset


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

IMAGECLEF2021_TUBERCULOSIS_CLASSES = {1:'Infiltrative', 2:'Focal', 3:'Tuberculoma', 4:'Miliary', 5:'Fibro-cavernous'}

class ImageClefTuberculosisDataset(Dataset):
    def __init__(self, root, split="train_all", get_CT = True, get_mask1 = True, get_mask2 = True, target_classes=(1,2,3,4,5), transform=None):
        self.root = root
        self.split = split
        self.get_CT = get_CT
        self.get_mask1 = get_mask1
        self.get_mask2 = get_mask2
        self.target_classes = target_classes
        self.transform = transform

        if self.split in ['train_all', 'train', 'val']:
            self.data_dir = os.path.join(self.root,"train")

        elif self.split == 'test':
            self.data_dir = os.path.join(self.root,"test")


        self.data_file = self.data_dir + "/Labelling data.csv"
        self.CT_dir = self.data_dir + "/ct"
        self.mask1_dir = self.data_dir + "/masks1"
        self.mask2_dir = self.data_dir + "/masks2"

        if self.split in ['train','val']:
            self.data_file = self.data_dir + f"/Labelling data_{self.split}.csv"

        if not os.path.exists(self.data_file):
            raise NotImplementedError("This dataset has not been added to the working environment yet")

        self.samples = pd.read_csv(self.data_file, sep=',')
        self.samples = self.samples.loc[self.samples['TypeOfTB'].isin(target_classes)]

        
    def __len__(self):
        return len(self.samples.index)

    def __getitem__(self, idx):
        CT = torch.BFloat16Tensor()
        z2xy = torch.BFloat16Tensor()
        mask1 = torch.BFloat16Tensor()
        mask2 = torch.BFloat16Tensor()
        
        if self.split in ['train_all', 'train', 'val']:
            Filename,TypeOfTB = self.samples.iloc[idx]
        else:
            Filename = self.samples["Filename"].iloc[idx]
            
        if self.get_CT:
            CT = nb.load(os.path.join(self.CT_dir,Filename))
            z2xy = abs(CT.affine[2, 2] / CT.affine[0, 0])
            CT = torch.BFloat16Tensor(CT.get_data()).unsqueeze(-1).permute(3,2,0,1)
        if self.get_mask1:
            mask1 = nb.load(os.path.join(self.mask1_dir,Filename))
            mask1 = torch.BFloat16Tensor(mask1.get_data()).unsqueeze(-1).permute(3,2,0,1)
        if self.get_mask2:
            mask2 = nb.load(os.path.join(self.mask2_dir,Filename))
            mask2 = torch.BFloat16Tensor(mask2.get_data()).unsqueeze(-1).permute(3,2,0,1)

        if self.transform:
            CT, mask1, mask2, z2xy = self.transform(CT, mask1, mask2, z2xy)

        if self.split in ['train_all', 'train', 'val']:
            return (CT, mask1, mask2, z2xy), TypeOfTB
        else:
            return CT, mask1, mask2, z2xy


datasets = {
    'CIFAR-10': torchvision.datasets.CIFAR10,
    'CIFAR-100': torchvision.datasets.CIFAR100,
    'ImageNet2012': torchvision.datasets.ImageFolder,
    'ImageCLEF2021-Tuberculosis': ImageClefTuberculosisDataset,
}