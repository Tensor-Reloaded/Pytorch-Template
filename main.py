import logging
import os
from functools import partial
from shutil import copyfile

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import configure, maybe_reset_seed, prepare_dataset_and_transforms, init_dataset, init_dataloader, \
    init_model, init_weights, init_batch_norm, load_model, init_optimizer, init_scheduler, init_criterion, \
    init_metrics, maybe_load_optimizer, metrics, register_metrics, EarlyStopping, tensorboard_export_dump, \
    print_metrics, get_batch_size, to_device, attr_is_valid, disable_bn, enable_bn


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: DictConfig) -> None:
    configure(config)
    Solver(config).run()


class Solver:
    def __init__(self, config: DictConfig):
        self.criterion = None
        self.output_transformations = None  # TODO: add them
        self.scheduler_name = None
        self.scheduler = None
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
        self.device_type = "cuda" if "cuda" in self.device else "cpu"

        self.test_batch_plot_idx = 0
        self.val_batch_plot_idx = 0
        self.train_batch_plot_idx = 0
        self.epoch = 1

        self.es = EarlyStopping(patience=self.args.es_patience, min_delta=self.args.es_min_delta)
        self.scaler = GradScaler(enabled=self.args.half and self.args.grad_scaler)  # FIXME this runs only on cuda

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

    def prepare_loader(self, loader):
        if self.args.progress_bar:
            loader = tqdm(loader)
        return loader

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
        self.criterion = init_criterion(self.args.loss, self.device, hasattr(self.args, "dba"))

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
            pass
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
        # TODO: Check and implement bellow
        print("infer:")
        self.model.eval()

        predictions = []
        filenames = []
        with torch.no_grad():
            for batch_num, (filename, data) in enumerate(self.infer_loader):
                if isinstance(data, list):
                    data = [i.to(self.device) for i in data]
                else:
                    data = data.to(self.device)

                with autocast(enabled=self.args.half):
                    output = self.model(data)
                    if self.output_transformations is not None:
                        output = self.output_transformations(output)

                if isinstance(output, list) or isinstance(output, tuple):
                    for pred_idx, o in enumerate(output):
                        o = o.cpu()
                        if len(predictions) <= pred_idx:
                            predictions.append(torch.tensor(o))
                        else:
                            predictions[pred_idx] = torch.cat((predictions[pred_idx], o))
                else:
                    output = output.cpu()
                    if isinstance(output, torch.Tensor):
                        predictions = output
                    else:
                        predictions = torch.tensor(output)

                filenames.extend(filename)

        return filenames, predictions

    def save_batch_metrics(self, output, target, metric_type):
        metrics_results = {}
        metrics_results = register_metrics(
            self.metrics, metric_type, "batch", metrics_results, prediction=output, target=target)
        if metric_type == "train":
            metrics_results = register_metrics(self.metrics, "solver", "batch", metrics_results, solver=self)
            batch_index = self.get_train_batch_plot_idx()
        elif metric_type == "val":  # val or test
            batch_index = self.get_val_batch_plot_idx()
        else:
            batch_index = self.get_test_batch_plot_idx()

        if len(metrics_results):
            print_metrics(self.writer, metrics_results, batch_index)

    def train_get_output(self, data, hidden):
        if hidden is None:
            output = self.model(data)
        else:
            output = self.model(data, hidden)
        if self.output_transformations is not None:
            output = self.output_transformations(output)
        return output

    def train_get_loss(self, output, target, is_train):
        if hasattr(self.args.model, 'returns_loss') and self.args.model.returns_loss:
            loss = output
        else:
            loss = self.criterion(output, target)
        if is_train:
            loss /= self.args.train_dataset.update_every
        return loss

    def train_get_output_and_loss(self, data, target, hidden, is_train=True):
        with autocast(enabled=self.args.half, device_type=self.device_type):
            output = self.train_get_output(data, hidden)
            return output, self.train_get_loss(output, target, is_train)

    def train_maybe_apply_grad_penalty(self, loss):
        # TODO: check self.args.optimizer.grad_penalty > 0.0
        if attr_is_valid(self.args.optimizer, "grad_penalty"):
            # Creates gradients
            scaled_grad_params = torch.autograd.grad(outputs=self.scaler.scale(loss),
                                                     inputs=self.model.parameters(), create_graph=True)

            # Creates unscaled grad_params before computing the penalty. scaled_grad_params are
            # not owned by any optimizer, so ordinary division is used instead of scaler.unscale_:
            inv_scale = 1. / self.scaler.get_scale()
            grad_params = [p * inv_scale for p in scaled_grad_params]

            # Computes the penalty term and adds it to the loss
            with autocast(device_type=self.device_type):
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + (grad_norm * self.args.optimizer.grad_penalty)
        return loss

    def train_maybe_do_batch_reply(self):
        if attr_is_valid(self.args.optimizer, "batch_replay"):
            found_inf = False
            for _, param in self.model.named_parameters():
                if not param.grad.isfinite.all():  # checking for nan or inf
                    found_inf = True
                    break
            if found_inf:
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if isinstance(self.args.optimizer.batch_replay, (int, float)):
                    self.args.optimizer.batch_replay -= 1
            else:
                return True
        else:
            return True
        return False

    def train_create_scaler_func(self, data, target):
        step_partial_func = partial(self.scaler.step)
        return self.train_maybe_use_sam(step_partial_func, data, target)

    def train_maybe_use_sam(self, step_partial_func, data, target):
        if attr_is_valid(self.args.optimizer, "use_SAM"):
            def sam_closure():
                disable_bn(self.model)
                while True:
                    with autocast(enabled=self.args.half, device_type=self.device_type):
                        output = self.model(data)
                        if self.output_transformations is not None:
                            output = self.output_transformations(output)

                        if hasattr(self.args.model, 'returns_loss') and self.args.model.returns_loss:
                            loss = output
                        else:
                            loss = self.criterion(output, target)
                        loss = loss / self.args.train_dataset.update_every

                    if self.args.optimizer.grad_penalty is not None and self.args.optimizer.grad_penalty > 0.0:
                        # Creates gradients
                        scaled_grad_params = torch.autograd.grad(outputs=self.scaler.scale(loss),
                                                                 inputs=self.model.parameters(), create_graph=True)

                        # Creates unscaled grad_params before computing the penalty. scaled_grad_params are
                        # not owned by any optimizer, so ordinary division is used instead of scaler.unscale_:
                        inv_scale = 1. / self.scaler.get_scale()
                        grad_params = [p * inv_scale for p in scaled_grad_params]

                        # Computes the penalty term and adds it to the loss
                        with autocast(device_type=self.device_type):
                            grad_norm = 0
                            for grad in grad_params:
                                grad_norm += grad.pow(2).sum()
                            grad_norm = grad_norm.sqrt()
                            loss = loss + (grad_norm * self.args.optimizer.grad_penalty)

                    self.scaler.scale(loss).backward()

                    if self.args.optimizer.batch_replay:
                        found_inf = False
                        for _, param in self.model.named_parameters():
                            if param.grad.isnan().any() or param.grad.isinf().any():
                                found_inf = True
                                break
                        if found_inf:
                            self.scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                            if isinstance(self.args.optimizer.batch_replay, (int, float)):
                                self.args.optimizer.batch_replay -= 1
                        else:
                            break
                    else:
                        break

                if self.train_batch_plot_idx % self.args.train_dataset.update_every == 0:
                    self.scaler.unscale_(self.optimizer)

                    if self.args.optimizer.max_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optimizer.max_norm)
                    enable_bn(self.model)

            step_partial_func = partial(step_partial_func, closure=sam_closure)
        return step_partial_func

    def train_maybe_clip_grad(self):
        if hasattr(self.args.optimizer, "max_norm") and self.args.optimizer.max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optimizer.max_norm)

    def train_maybe_step_scheduler(self):
        if self.scheduler_name == "OneCycleLR":
            self.scheduler.step()

    def train_maybe_do_update(self, data, target):
        if self.train_batch_plot_idx % self.args.train_dataset.update_every == 0:
            self.scaler.unscale_(self.optimizer)
            self.train_maybe_clip_grad()

            step_partial_func = self.train_create_scaler_func(data, target)
            step_partial_func(self.optimizer)
            # self.model.update_moving_average()

            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)
            self.train_maybe_step_scheduler()

    def train_maybe_init_hidden(self, batch_size):
        if hasattr(self.model, "init_hidden"):
            hidden = self.model.init_hidden(batch_size, self.device)
        else:
            hidden = None
        return hidden

    def simple_train(self):
        # The reference method for training
        logging.info("train:\n")
        self.model.train()
        predictions = []
        targets = []
        loss_sum = 0.0

        for data, target in self.prepare_loader(self.train_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            output = self.model(data)
            loss = self.criterion(output, target)

            loss_sum += loss.item()
            loss.backward()

            self.train_maybe_clip_grad()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            predictions.extend(output.detach().cpu())
            targets.extend(target.cpu())

        return {
            "prediction": torch.stack(predictions) if len(predictions) else predictions,
            "target": torch.stack(targets) if len(targets) else targets,
            "loss": loss_sum / len(self.train_loader),
        }

    def train(self):
        logging.info("train:\n")
        self.model.train()

        predictions = []
        targets = []
        loss_sum = 0.0

        batch_size = get_batch_size(self.train_loader)
        hidden = self.train_maybe_init_hidden(batch_size)  # TODO: Support hidden initialization at each batch

        for data, target in self.prepare_loader(self.train_loader):
            data = to_device(data, self.device)
            target = to_device(target, self.device)

            while True:
                output, loss = self.train_get_output_and_loss(data, target, hidden)

                loss = self.train_maybe_apply_grad_penalty(loss)

                self.scaler.scale(loss).backward()

                if self.train_maybe_do_batch_reply():
                    break

            self.train_maybe_do_update(data, target)

            predictions.extend(output.detach().cpu())
            targets.extend(target.cpu())

            self.save_batch_metrics(output, target, "train")

        return {
            "prediction": torch.stack(predictions) if len(predictions) else predictions,
            "target": torch.stack(targets) if len(targets) else targets,
            "loss": loss_sum / len(self.train_loader),
        }

    @torch.no_grad()
    def val(self, do_test):
        self.model.eval()

        if do_test:
            metric_type = "test"
            loader = self.test_loader
        else:
            metric_type = "val"
            loader = self.val_loader

        logging.info(f"{metric_type}:\n")
        predictions = []
        targets = []
        loss_sum = 0.0

        batch_size = get_batch_size(loader)
        hidden = self.train_maybe_init_hidden(batch_size)

        for data, target in self.prepare_loader(loader):
            data = to_device(data, self.device)
            target = to_device(target, self.device)

            output, loss = self.train_get_output_and_loss(data, target, hidden, is_train=False)

            predictions.extend(output)
            targets.extend(target)
            loss_sum += loss.item()

            self.save_batch_metrics(output, target, metric_type)

        return {
            "prediction": torch.stack(predictions) if len(predictions) else predictions,
            "target": torch.stack(targets) if len(targets) else targets,
            "loss": loss_sum / len(loader),
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

    def get_test_batch_plot_idx(self):
        ret = self.test_batch_plot_idx
        self.test_batch_plot_idx += 1
        return ret

    def save_backup(self):
        raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()
