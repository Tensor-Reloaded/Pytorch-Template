import logging
import os
from shutil import copyfile

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import configure, maybe_reset_seed, prepare_dataset_and_transforms, init_dataset, init_dataloader, \
    init_model, init_weights, init_batch_norm, load_model, init_optimizer, init_scheduler, init_criterion, \
    init_metrics, maybe_load_optimizer, metrics, register_metrics, EarlyStopping, tensorboard_export_dump, print_metrics


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: DictConfig) -> None:
    configure(config)
    Solver(config).run()


class Solver:
    def __init__(self, config: DictConfig):
        self.scheduler_name = None
        self.scheduler = None
        self.val_batch_plot_idx = None
        self.train_batch_plot_idx = None
        self.epoch = 1
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

        self.es = EarlyStopping(patience=self.args.es_patience, min_delta=self.args.es_min_delta)

        if not self.args.save_dir:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir="runs/" + self.args.save_dir)

    def init(self):
        maybe_reset_seed(self.args.seed)
        self.init_dataset()
        self.init_model()
        self.init_optimizer()
        self.init_scheduler()
        self.init_criterion()
        self.init_metrics()

        self.maybe_load_state()

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
        self.optimizer = maybe_load_optimizer(self.optimizer, self.args.load_optimizer, self.args.restart_from_backup)

    def init_scheduler(self):
        # TODO: Implement many schedulers (list of schedulers)
        self.scheduler, self.scheduler_name = init_scheduler(self.args.scheduler, self.optimizer)

    def init_criterion(self):
        init_criterion(self.args.loss, self.device, hasattr(self.args, "dba"))

    def init_metrics(self):
        self.metrics = init_metrics(self.args)

    def maybe_load_state(self):
        if len(self.args.load_training_state) > 0:
            training_state = torch.load(self.args.load_training_state)
            self.epoch = training_state["epoch"]
            self.train_batch_plot_idx = training_state["train_batch_plot_idx"]
            self.val_batch_plot_idx = training_state["val_batch_plot_idx"]
            for metric in self.metrics['solver']['epoch']:
                if metric.name == "Real Epoch Count":
                    metric.metric_func.counter = training_state["real_epoch_count"]
                    break

    def maybe_infer(self):
        # TODO: Rewrite for general case
        if self.args.infer_only:
            filenames, predictions = self.infer()
            predictions = predictions.argmax(-1) + 1
            save_path = os.path.join(self.save_dir, "predictions.csv")
            pd.DataFrame({'Patient': filenames, 'Class': predictions.cpu().numpy()}).to_csv(save_path, header=False,
                                                                                            index=False)
            exit()

    def maybe_register_best(self, metrics_results, best_metrics):
        if self.args.optimized_metric in metrics_results:
            save_best_metric = False

            if self.args.optimized_metric not in best_metrics:
                best_metrics[self.args.optimized_metric] = metrics_results[self.args.optimized_metric]
                save_best_metric = True
            if metrics[self.args.optimized_metric.split('/')[-1]]['higher_is_better']:
                if best_metrics[self.args.optimized_metric] < metrics_results[self.args.optimized_metric]:
                    best_metrics[self.args.optimized_metric] = metrics_results[self.args.optimized_metric]
                    save_best_metric = True
            else:
                if best_metrics[self.args.optimized_metric] > metrics_results[self.args.optimized_metric]:
                    best_metrics[self.args.optimized_metric] = metrics_results[self.args.optimized_metric]
                    save_best_metric = True

            if save_best_metric and self.args.save_model:
                best = best_metrics[self.args.optimized_metric]
                self.save(self.epoch, best)
                logging.info(f"===> BEST {self.args.optimized_metric} PERFORMANCE: {best:.5f}")

    def maybe_save_model(self):
        if self.args.save_model and self.epoch % self.args.save_interval == 0:
            self.save_backup()
            self.save(self.epoch, 0)

    def scheduler_step(self, metrics_results):
        if self.scheduler_name == "MultiStepLR":
            self.scheduler.step()
        elif self.scheduler_name == "ReduceLROnPlateau":
            self.scheduler.step(metrics_results[self.args.scheduler_metric])
        elif self.scheduler_name == "OneCycleLR":
            pass  # FIXME
        else:
            self.scheduler.step()

    def maybe_early_stopping(self, metrics_results):
        if self.es.step(metrics_results[self.args.es_metric]):
            print("Early stopping")
            raise KeyboardInterrupt

    def end_training(self, best_metrics):
        if self.args.optimized_metric not in best_metrics:
            best = torch.nan
        else:
            best = best_metrics[self.args.optimized_metric]
        logging.info(f"===> BEST {self.args.optimized_metric} PERFORMANCE: {best:.5f}")

        files = os.listdir(self.save_dir)
        paths = [os.path.join(self.save_dir, basename) for basename in files if "_0" not in basename]
        if len(paths) > 0:
            src = max(paths, key=os.path.getctime)
            copyfile(src, os.path.join("runs", self.args.save_dir, os.path.basename(src)))

        with open("runs/" + self.args.save_dir + "/README.md", 'a+') as f:
            f.write(f"\n## {self.args.optimized_metric}\n {best_metrics[self.args.optimized_metric]:.5f}")
        tensorboard_export_dump(self.writer)
        logging.info("Saved best accuracy checkpoint")

        return best_metrics[self.args.optimized_metric]

    def save(self, epoch, metric, tag=None):
        if tag is not None:
            tag = "_" + tag
        else:
            tag = ""
        model_out_path = os.path.join(self.save_dir, f"model_{epoch}_{metric}{tag}.pth")
        optimizer_out_path = os.path.join(self.save_dir, f"optimizer_{epoch}_{metric}{tag}.pth")
        training_state_out_path = os.path.join(self.save_dir, f"training_state_{epoch}_{metric}{tag}.pth")
        torch.save(self.model.state_dict(), model_out_path)
        torch.save(self.optimizer.state_dict(), optimizer_out_path)

        training_state = {
            "epoch": self.epoch,
            "batch_size": self.args.train_dataset.batch_size,
            "train_batch_plot_idx": self.train_batch_plot_idx,
            "val_batch_plot_idx": self.val_batch_plot_idx,
        }

        for metric in self.metrics['solver']['epoch']:
            if metric.name == "Real Epoch Count":
                training_state["real_epoch_count"] = metric.metric_func.counter
                break

        torch.save(training_state, training_state_out_path)

        logging.info(f"Checkpoint saved to {model_out_path}")

    def infer(self):
        raise NotImplementedError()

    def save_batch_metrics(self, output, target):
        metrics_results = {}
        metrics_results = register_metrics(
            self.metrics, "train", "batch", metrics_results, prediction=output, target=target)
        metrics_results = register_metrics(self.metrics, "solver", "batch", metrics_results, solver=self)
        print_metrics(self.writer, metrics_results, self.get_train_batch_plot_idx())

    def train(self):
        logging.info("train:\n")

        predictions = []
        targets = []
        loss_sum = 0.0

        # batch_size = get_batch_size(self.train_loader)
        # I_N = torch.eye(batch_size, device=self.device)

        # if hasattr(self.model, "init_hidden"):
        #     hidden = self.model.init_hidden(batch_size, self.device)

        train_loader = self.train_loader
        if self.args.progress_bar:
            train_loader = tqdm(train_loader)

        for data, target in train_loader:
            output = None

            predictions.extend(output.detach().cpu())
            targets.extend(target.cpu())

            self.save_batch_metrics(output, target)


        return {
            "prediction": torch.stack(predictions) if len(predictions) else predictions,
            "target": torch.stack(targets) if len(targets) else targets,
            "loss": loss_sum,
        }

    def run(self):
        self.init()
        try:
            self.maybe_infer()
            best_metrics = {}

            while self.epoch < self.args.epochs:
                logging.info(f"\n===> epoch: {self.epoch}/{self.args.epochs}")

                train_results = self.train()

                metrics_results = {}
                metrics_results = register_metrics(self.metrics, "train", "epoch", metrics_results, **train_results)

                if self.val_loader is not None and self.epoch % self.args.val_every == 0:
                    val_results = self.val(do_test=False)
                    metrics_results = register_metrics(self.metrics, "val", "epoch", metrics_results, **val_results)

                if self.test_loader is not None and self.epoch % self.args.test_every == 0:
                    test_results = self.val(do_test=True)
                    metrics_results = register_metrics(self.metrics, "test", "epoch", metrics_results, **test_results)

                metrics_results = register_metrics(self.metrics, "solver", "epoch", metrics_results, solver=self)

                print_metrics(self.writer, metrics_results, self.epoch)

                self.maybe_register_best(metrics_results, best_metrics)
                self.maybe_save_model()
                self.scheduler_step(metrics_results)
                self.maybe_early_stopping(metrics_results)

        except KeyboardInterrupt:
            pass
        self.end_training(best_metrics)

    def get_train_batch_plot_idx(self):
        ret = self.train_batch_plot_idx
        self.train_batch_plot_idx += 1
        return ret

    def get_val_batch_plot_idx(self):
        ret = self.val_batch_plot_idx
        self.val_batch_plot_idx += 1
        return ret

    def save_backup(self):
        raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()
