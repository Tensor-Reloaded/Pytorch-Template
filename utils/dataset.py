from __future__ import annotations
# TODO: Future absolut import
import logging
import random
from functools import cached_property, cache

import numpy as np
import torchvision
from torchvision import transforms
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from .transformations import init_transforms
from .utils import attr_is_valid


def prepare_dataset_and_transforms(dataset):
    if dataset.name not in datasets:
        logging.error(f"Dataset {dataset.name} not implemented!")
        exit()
    cached_transforms, runtime_transforms = None, None
    if hasattr(dataset, "transform"):
        cached_transforms, runtime_transforms = init_transforms(dataset.transform)

    return dataset, cached_transforms, runtime_transforms


def init_dataset(dataset_config, cached_transforms, runtime_transforms, device) -> DatasetWrapper:
    parameters = OmegaConf.to_container(dataset_config.load_params, resolve=True)
    parameters = {k: v for k, v in parameters.items() if v is not None}
    if "device" in parameters and "cuda" in parameters["device"]:
        parameters["device"] = device

    dataset = datasets[dataset_config.name](**parameters)
    if attr_is_valid(dataset_config, "subset") and dataset_config.subset != '' and dataset_config.subset >= 0:
        dataset = select_dataset_subset(dataset, dataset_config.subset)

    if attr_is_valid(dataset_config, "corrupt") and attr_is_valid(dataset_config, "corrupt_subset"):
        dataset = corrupt_dataset(dataset, dataset_config.corrupt_subset)

    dataset = DatasetWrapper(dataset=dataset,
                             cached_transforms=cached_transforms,
                             runtime_transforms=runtime_transforms,
                             save_in_memory=dataset_config.save_in_memory)

    return dataset


def select_dataset_subset(dataset, subset: float) -> tuple:
    if subset < 1.0:
        ix_size = int(subset * len(dataset))
    else:
        ix_size = int(subset)
    indices = np.random.choice(len(dataset), size=ix_size, replace=False)

    dataset_subset = []
    for idx in indices:
        dataset_subset.append(dataset[idx])

    return tuple(dataset_subset)


def corrupt_dataset(dataset, corrupt_subset: float) -> tuple:
    if corrupt_subset <= 0.0:
        return dataset
    if corrupt_subset < 1.0:
        corrupt_size = int(corrupt_subset * len(dataset))
    else:
        corrupt_size = int(corrupt_subset)
    dataset = list(dataset)  # mutable sequence
    random.shuffle(dataset)

    new_labels = np.random.randint(0, 10, corrupt_size)
    for i in range(corrupt_subset):
        dataset[i] = dataset[i][0], new_labels[i]

    return tuple(dataset)


def transform_dataset(dataset, transforms: transforms.Compose) -> tuple:
    dataset_copy = []

    for idx in range(len(dataset)):
        tensor, target = dataset[idx]
        tensor, target = transforms((tensor, target))
        dataset_copy.append((tensor, target))

    return tuple(dataset_copy)


class DatasetWrapper(Dataset):
    def __init__(self, dataset, cached_transforms=None, runtime_transforms=None, save_in_memory=False):
        self.dataset = dataset

        if cached_transforms is None:
            cached_transforms = []
        if runtime_transforms is None:
            runtime_transforms = []

        if save_in_memory:
            self.dataset = transform_dataset(self.dataset, transforms.Compose(cached_transforms))
        else:
            runtime_transforms = cached_transforms + runtime_transforms

        self.runtime_transforms = transforms.Compose(runtime_transforms)

    @cache  # Maybe a bit overkill, we should cache the length ourselves.
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):  # This could be @torch.compiled, but check for improvement first
        return self.runtime_transforms(self.dataset[index])


class ComposedDataset(Dataset):
    def __init__(self, dataset_dict):
        # TODO instead of dataset_dict, pass a list of dataset names and read their config files directly
        self.dataset_list = []
        for dataset_name, dataset_params in dataset_dict.items():
            if dataset_name not in datasets:
                print(f"This dataset is not implemented ({dataset_name}), go ahead and commit it")
                exit()

            dataset = datasets[dataset_name](**dataset_params)
            self.dataset_list.append(dataset)

    def __len__(self):
        return max([len(d) for d in self.dataset_list])

    def __getitem__(self, idx):
        tensors = []
        targets = []
        for d in self.dataset_list:
            tensor, target = d[idx % len(d)]
            tensors.append(tensor)
            targets.append(target)
        return tensors, targets


datasets = {
    'CIFAR-10': torchvision.datasets.CIFAR10,
    'CIFAR-100': torchvision.datasets.CIFAR100,
    'ImageNet2012': torchvision.datasets.ImageFolder,
    'MNIST': torchvision.datasets.MNIST,
    'ComposedDataset': ComposedDataset
}
