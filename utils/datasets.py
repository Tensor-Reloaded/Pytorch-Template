import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

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

datasets = {
    'CIFAR-10': torchvision.datasets.CIFAR10,
    'CIFAR-100': torchvision.datasets.CIFAR100,
    'ImageNet2012': torchvision.datasets.ImageFolder,
    'MNIST': torchvision.datasets.MNIST,
}


def select_dataset_subset(dataset, subset):
    if subset < 1.0:
        ix_size = int(subset * len(dataset))
    else:
        ix_size = int(subset)
    indices = np.random.choice(len(dataset), size=ix_size, replace=False)

    dataset_subset = []
    for idx in indices:
        dataset_subset.append(dataset[idx])

    return dataset_subset


class MemoryStoredDataset(Dataset):
    def __init__(self, dataset, transformations_cached=None, transformations_not_cached=None, save_in_memory=False):
        self.dataset = dataset

        if transformations_cached is None:
            transformations_cached = []
        if transformations_not_cached is None:
            transformations_not_cached = []

        save_in_memory = save_in_memory and len(transformations_cached) > 0
        if not save_in_memory:
            self.transformations = transforms.Compose(transformations_cached + transformations_not_cached)
        else:
            self.cache_transformations(transforms.Compose(transformations_cached))
            self.transformations = transforms.Compose(transformations_not_cached)

    def cache_transformations(self, transformations):
        dataset_copy = []

        for idx in range(len(self.dataset)):
            tensor, target = self.dataset[idx]
            for t in transformations.transforms:
                tensor, target = t((tensor, target))
            dataset_copy.append((tensor, target))

        self.dataset = dataset_copy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tensor, target = self.dataset[idx]
        for t in self.transformations.transforms:
            tensor, target = t((tensor, target))
        return tensor, target


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


datasets['ComposedDataset'] = ComposedDataset

if __name__ == '__main__':
    pass
