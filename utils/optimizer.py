import logging

import torch
import torch_optimizer
from omegaconf import OmegaConf
from torch import optim


def init_optimizer(optimizer_config, model):
    parameters = OmegaConf.to_container(optimizer_config.parameters, resolve=True)
    parameters = {k: v for k, v in parameters.items() if v is not None}
    parameters["params"] = model.parameters()

    try:
        optimizer = getattr(torch_optimizer, optimizer_config.name)
    except AttributeError:
        try:
            optimizer = getattr(optim, optimizer_config.name)
        except AttributeError:
            try:
                optimizer = optimizers[optimizer_config.name]
            except KeyError:
                logging.error(f"Optimizer {optimizer_config.name} does not exist!")
                exit()

    if hasattr(optimizer_config, "use_SAM") and optimizer_config.use_SAM:
        optimizer = optimizers['SAM'](params=parameters["params"], base_optimizer=optimizer,
                                      rho=optimizer_config.SAM_rho)
    else:
        optimizer = optimizer(**parameters)

    if hasattr(optimizer_config, "use_lookahead") and optimizer_config.use_lookahead:
        optimizer = torch_optimizer.Lookahead(optimizer, k=optimizer_config.lookahead_k,
                                              alpha=optimizer_config.lookahead_alpha)

    return optimizer


def maybe_load_optimizer(optimizer, optimizer_path, restart_from_backup):
    if len(optimizer_path) > 0 and restart_from_backup:
        optimizer.load_state_dict(torch.load(optimizer_path))
        logging.info(f"Loaded optimizer from {optimizer_path}")
    return optimizer


# TODO: use library one
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.5, adaptive=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


optimizers = {
    'SAM': SAM
}
