import logging
import os

import hydra
from omegaconf import DictConfig

from utils import configure, maybe_reset_seed, prepare_dataset_and_transforms, init_dataset, init_dataloader, \
    init_model, init_weights, init_batch_norm, load_model, init_optimizer, init_scheduler, init_criterion, init_metrics


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: DictConfig) -> None:
    configure(config)
    Solver(config).run()


class Solver:
    def __init__(self, config: DictConfig):
        self.metrics = None
        self.optimizer = None
        self.save_dir = None
        self.infer_loader = None
        self.infer_set = None
        self.test_loader = None
        self.test_set = None
        self.val_set = None
        self.val_loader = None
        self.train_loader = None
        self.train_set = None
        self.train_set = None
        self.model = None
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

    def init_model(self):
        self.model = init_model(self.args.model)

        self.save_dir = os.path.join(self.args.storage_dir, "model_weights", self.args.save_dir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        init_weights(self.model, self.args.initialization)
        if self.args.initialization_batch_norm:
            init_batch_norm(self.model)
        if len(self.args.load_model):
            self.model = load_model(self.args.model, self.args.load_model, self.model, self.device)

        self.model = self.model.to(self.device)

    def init_optimizer(self):
        self.optimizer = init_optimizer(self.args.optimizer, self.model)

    def init_scheduler(self):
        # TODO: Implement many schedulers (list of schedulers)
        init_scheduler(self.args.scheduler, self.optimizer)

    def init_criterion(self):
        init_criterion(self.args.loss, self.device, hasattr(self.args, "dba"))

    def init_metrics(self):
        self.metrics = init_metrics(self.args)

    def run(self):
        self.init()


if __name__ == "__main__":
    main()
