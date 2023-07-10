# TODO: Add type hints
import os

from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def init_dataloader(dataset_config, dataset, pin_memory_device) -> DataLoader:
    if dataset_config.batch_size == 'None':
        # This means that we batch in our dataset
        dataset_config.batch_size = None
        collate_fn = lambda x: x
        dataset_config.drop_last = False
    else:
        collate_fn = None

    # TODO: On collate_fn side we could implement batch transformations (applied at batch level, not at sample level).
    # This requires implementing them. Might have impact if transformations are costly and not many workers are
    # available.

    sampler = None

    if os.name == 'nt':
        dataset_config.num_workers = 0

    has_cuda = "cuda" in pin_memory_device
    on_cuda = hasattr(dataset_config.load_params, 'device') and 'cuda' in dataset_config.load_params.device and has_cuda
    pin_memory = not on_cuda and dataset_config.pin_memory and has_cuda
    if not pin_memory:
        pin_memory_device = ""

    OmegaConf.set_struct(dataset_config, False)  # Allows write
    if not hasattr(dataset_config, "persistent_workers"):
        dataset_config.persistent_workers = False

    return DataLoader(
        dataset=dataset, batch_size=dataset_config.batch_size, shuffle=(dataset_config.shuffle and sampler is None),
        sampler=sampler, num_workers=dataset_config.num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
        pin_memory_device=pin_memory_device, drop_last=dataset_config.drop_last,
        persistent_workers=dataset_config.persistent_workers)  # !~Persistent workers may use a lot of ram.


def get_batch_size(loader):
    if loader.batch_sampler is None:
        if hasattr(loader.dataset, "dataset") and hasattr(loader.dataset.dataset, "batch_size"):
            return loader.dataset.dataset.batch_size

        raise NotImplementedError("Not implemented batch size for " + str(loader))
    return loader.batch_sampler.batch_size
