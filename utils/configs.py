from __future__ import annotations
from fnmatch import fnmatch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple, Sequence
import logging
import os
import yaml
import zipfile


def configure(config: DictConfig) -> None:
    config = change_save_dir(config)
    save_configuration(config)


def change_save_dir(config: DictConfig) -> DictConfig:
    head, tail = os.path.split(config.save_dir)
    if tail == 'None':  # I am not sure when this happens, and whether this happens at all
        logging.warning(f"This is when it happens: head is {head}, tail is {tail}, save dir is {config.save_dir}")
        config.save_dir = head
    else:
        config.save_dir = os.path.join(head, tail)
    return config


def save_configuration(config: DictConfig) -> None:
    save_config_path = os.path.join(os.getcwd(), "runs", config.save_dir)
    os.makedirs(save_config_path, exist_ok=True)
    with open(os.path.join(save_config_path, "README.yaml"), 'w+') as f:
        yaml.dump(OmegaConf.to_yaml(config, resolve=True), f, default_flow_style=False)
    save_current_code(save_config_path)


def save_current_code(path: str) -> None:
    logging.info(f"Saving current code to {path}")

    project_root = get_original_cwd()
    files = get_all_files(project_root)
    filters = create_file_filter(os.path.join(project_root, ".gitignore"))
    for ignore_filter in filters:
        files = [x for x in files if not match_file(x.replace(project_root, ''), ignore_filter)]

    make_zip(os.path.join(path, "files.zip"), project_root, files)


def create_file_filter(gitignore_path: str | None = None) -> Tuple[str]:
    ignore_files = []
    if gitignore_path is not None and os.path.exists(gitignore_path):
        # If .gitignore exists, use .gitignore
        def preprocess_line(line: str) -> str:
            line = line.rstrip('\n')  # remove \n from end
            return line.split("#")[0]  # remove everything after comment

        with open(gitignore_path, "r") as f:
            lines = map(preprocess_line, f.readlines())
            lines = filter(lambda x: len(x) > 0, lines)  # filter empty lines

        ignore_files.extend(lines)

    ignore_files += [
        "venv/*", "multirun/*", "__pycache__/*", ".git/*", "data/*", "results/*", "*.txt", "*.md", 'outputs/*', '.git*'
    ]
    return tuple(set(ignore_files))


# TODO: Move in file utils
def get_all_files(path: str) -> List[str]:
    all_files = []
    for root, _, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def match_file(file: str, pattern: str) -> bool:
    if file.startswith("/") and not pattern.startswith("/"):
        file = file[1:]
    if file.startswith("\\") and not pattern.startswith("\\"):
        file = file[1:]
    return fnmatch(file, pattern)


def make_zip(zip_path: str, project_root: str, files: Sequence[str]) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for file in files:
            z.write(file, file.replace(project_root, ''))
