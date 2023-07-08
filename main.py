import logging

import hydra
from omegaconf import DictConfig

from utils import configure, maybe_reset_seed, prepare_dataset_and_transforms, init_dataset, init_dataloader


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: DictConfig) -> None:
    configure(config)
    Solver(config).run()


class Solver:
    def __init__(self, config: DictConfig):
        self.infer_loader = None
        self.infer_set = None
        self.test_loader = None
        self.test_set = None
        self.val_set = None
        self.val_loader = None
        self.train_loader = None
        self.train_set = None
        self.train_set = None
        self.args = config
        self.device = self.args.device

    def init(self):
        maybe_reset_seed(self.args.seed)
        self.init_dataset()
        self.init_model()
        self.init_optimizer()
        self.init_scheduler()
        self.init_criterion()
        self.init_metrics()

    def get_set_and_loader(self, set_name: str):
        if hasattr(self.args, set_name):
            logging.info(f"Loading {set_name}!")
            dataset_config, cached_transforms, runtime_transforms = prepare_dataset_and_transforms(
                getattr(self.args, set_name))
            dataset = init_dataset(dataset_config, cached_transforms, runtime_transforms, self.device)
            loader = init_dataloader(dataset_config, dataset, self.device)
            return dataset, loader
        return None, None

    def init_dataset(self):
        self.train_set, self.train_loader = self.get_set_and_loader("train_dataset")
        self.val_set, self.val_loader = self.get_set_and_loader("val_dataset")
        self.test_set, self.test_loader = self.get_set_and_loader("test_dataset")
        self.infer_set, self.infer_loader = self.get_set_and_loader("infer_dataset")


    def run(self):
        self.init()


if __name__ == "__main__":
    main()
