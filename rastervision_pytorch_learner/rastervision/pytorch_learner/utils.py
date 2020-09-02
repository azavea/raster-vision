from typing import Tuple, Optional
from pathlib import Path
import os.path
import shutil
from glob import glob

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.hub
import numpy as np
from PIL import ImageColor

from rastervision.pipeline.file_system import (download_if_needed, unzip)


def color_to_triple(color: Optional[str] = None) -> Tuple[int, int, int]:
    """Given a PIL ImageColor string, return a triple of integers
    representing the red, green, and blue values.

    If color is None, return a random color.

    Args:
         color: A PIL ImageColor string

    Returns:
         An triple of integers

    """
    if color is None:
        r = np.random.randint(0, 0x100)
        g = np.random.randint(0, 0x100)
        b = np.random.randint(0, 0x100)
        return (r, g, b)
    else:
        return ImageColor.getrgb(color)


def compute_conf_mat(out, y, num_labels):
    labels = torch.arange(0, num_labels).to(out.device)
    return ((out == labels[:, None]) & (y == labels[:, None, None])).sum(
        dim=2, dtype=torch.float32)


def compute_conf_mat_metrics(conf_mat, label_names, eps=1e-6):
    # eps is to avoid dividing by zero.
    eps = torch.tensor(eps)
    conf_mat = conf_mat.cpu()
    gt_count = conf_mat.sum(dim=1)
    pred_count = conf_mat.sum(dim=0)
    total = conf_mat.sum()
    true_pos = torch.diag(conf_mat)
    precision = true_pos / torch.max(pred_count, eps)
    recall = true_pos / torch.max(gt_count, eps)
    f1 = (2 * precision * recall) / torch.max(precision + recall, eps)

    weights = gt_count / total
    weighted_precision = (weights * precision).sum()
    weighted_recall = (weights * recall).sum()
    weighted_f1 = ((2 * weighted_precision * weighted_recall) / torch.max(
        weighted_precision + weighted_recall, eps))

    metrics = {
        'avg_precision': weighted_precision.item(),
        'avg_recall': weighted_recall.item(),
        'avg_f1': weighted_f1.item()
    }
    for ind, label in enumerate(label_names):
        metrics.update({
            '{}_precision'.format(label): precision[ind].item(),
            '{}_recall'.format(label): recall[ind].item(),
            '{}_f1'.format(label): f1[ind].item(),
        })
    return metrics


class AlbumentationsDataset(Dataset):
    """An adapter to use arbitrary datasets with albumentations transforms."""

    def __init__(self, orig_dataset, transform=None):
        """Constructor.

        Args:
            orig_dataset: (Dataset) which is assumed to return PIL Image objects
                and not perform any transforms of its own
            transform: (albumentations.core.transforms_interface.ImageOnlyTransform)
        """
        self.orig_dataset = orig_dataset
        self.transform = transform

    def __getitem__(self, ind):
        x, y = self.orig_dataset[ind]
        x = np.array(x)
        if self.transform:
            x = self.transform(image=x)['image']
        x = torch.tensor(x).permute(2, 0, 1).float() / 255.0
        return x, y

    def __len__(self):
        return len(self.orig_dataset)


class SplitTensor(nn.Module):
    ''' Wrapper around `torch.split` '''

    def __init__(self, size_or_sizes, dim):
        super().__init__()
        self.size_or_sizes = size_or_sizes
        self.dim = dim

    def forward(self, X):
        return X.split(self.size_or_sizes, dim=self.dim)


class Parallel(nn.ModuleList):
    ''' Passes inputs through multiple `nn.Module`s in parallel.
        Returns a tuple of outputs.
    '''

    def __init__(self, *args):
        super().__init__(args)

    def forward(self, xs):
        if isinstance(xs, torch.Tensor):
            return tuple(m(xs) for m in self)
        assert len(xs) == len(self)
        return tuple(m(x) for m, x in zip(self, xs))


class AddTensors(nn.Module):
    ''' Adds all its inputs together. '''

    def forward(self, xs):
        return sum(xs)


def _repo_name_to_dir(repo: str, hub_dir: str):
    from torch.hub import _parse_repo_info
    repo_owner, repo_name, branch = _parse_repo_info(repo)
    normalized_br = branch.replace('/', '_')
    dir_name = '_'.join([repo_owner, repo_name, normalized_br])
    repo_dir = os.path.join(hub_dir, dir_name)
    return repo_dir


def _uri_to_dir(uri: str, hub_dir: str) -> str:
    hubconf_dir = Path(hub_dir) / Path(uri).stem
    return str(hubconf_dir)


def get_hubconf_dir_from_cfg(cfg, hub_dir: str):
    if cfg.name is not None:
        return os.path.join(hub_dir, cfg.name)
    if cfg.github_repo is not None:
        return _repo_name_to_dir(cfg.github_repo, hub_dir)
    return _uri_to_dir(cfg.uri, hub_dir)


def torch_hub_load_github(repo: str, hub_dir: str, model: str, hubconf_dir: str = None, *args,
                          **kwargs):
    torch.hub.set_dir(hub_dir)
    model = torch.hub.load(github=repo, model=model, *args, **kwargs)
    if hubconf_dir is not None:
        shutil.move(_repo_name_to_dir(repo, hub_dir), hubconf_dir)
    return model


def torch_hub_load_uri(uri: str, hubconf_dir: str, model: str, tmp_dir: str,
                       *args, **kwargs):
    is_zip = Path(uri).suffix.lower() == '.zip'
    if is_zip:
        zip_path = download_if_needed(uri, tmp_dir)
        unzip_dir = os.path.join(tmp_dir, '_staging')
        if os.path.isdir(unzip_dir):
            shutil.rmtree(unzip_dir)
        unzip(zip_path, target_dir=unzip_dir)

        if os.path.isdir(hubconf_dir):
            shutil.rmtree(hubconf_dir)

        contents = list(glob(f'{unzip_dir}/*'))
        if (len(contents) == 1) and os.path.isdir(contents[0]):
            sub_dir = contents[0]
            shutil.move(sub_dir, hubconf_dir)
        else:
            shutil.move(unzip_dir, hubconf_dir)
    else:
        shutil.copytree(uri, hubconf_dir)

    model = torch_hub_load_local(hubconf_dir, model, *args, **kwargs)
    return model


def torch_hub_load_local(hubconf_dir: str, model: str, *args, **kwargs):
    from torch.hub import (sys, import_module, MODULE_HUBCONF,
                           _load_entry_from_hubconf)

    verbose = kwargs.get('verbose', True)
    kwargs.pop('verbose', None)

    sys.path.insert(0, hubconf_dir)

    hub_module = import_module(MODULE_HUBCONF,
                               os.path.join(hubconf_dir, MODULE_HUBCONF))

    entry = _load_entry_from_hubconf(hub_module, model)
    model = entry(*args, **kwargs)

    sys.path.remove(hubconf_dir)

    return model
