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

datasets = {
    'CIFAR-10': torchvision.datasets.CIFAR10,
    'CIFAR-100': torchvision.datasets.CIFAR100,
    'ImageNet2012': torchvision.datasets.ImageFolder,
    'MNIST': torchvision.datasets.MNIST,
}


class MemoryStoredDataset(Dataset):
    def __init__(self, dataset, transformations=None, save_in_memory=False, cache_index=0):
        self.dataset = dataset
        self.transformations = transformations

        self.cache_index = cache_index
        self.save_in_memory = save_in_memory
        if not self.save_in_memory:
            self.cache_index = 0

        if self.cache_index is None or self.cache_index == 0:
            self.save_in_memory = False

        if self.save_in_memory:
            self.stuff_in_memory = []
            for idx in range(len(self.dataset)):
                tensor, target = self.dataset[idx]
                for t in self.transformations.transforms[:self.cache_index]:
                    tensor, target = t((tensor, target))
                self.stuff_in_memory.append((tensor, target))
            self.getitem = self.getitem_cached
        else:
            self.getitem = self.getitem_not_cached

    def __len__(self):
        return len(self.dataset)

    def getitem_cached(self, idx):
        tensor, target = self.stuff_in_memory[idx]
        for t in self.transformations.transforms[self.cache_index:]:
            tensor, target = t((tensor, target))
        return tensor, target

    def getitem_not_cached(self, idx):
        tensor, target = self.dataset[idx]
        for t in self.transformations.transforms:
            tensor, target = t((tensor, target))
        return tensor, target

    def __getitem__(self, idx):
        return self.getitem(idx)


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
