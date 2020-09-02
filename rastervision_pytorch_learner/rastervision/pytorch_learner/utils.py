from typing import Tuple, Optional, Any
from pathlib import Path
from os.path import join, isdir, splitext
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


def _remove_dir(path):
    """ Remove a directory if it exists. """
    if isdir(path):
        shutil.rmtree(path)


def _repo_name_to_dir_name(repo: str) -> str:
    """Adapted from torch.hub._get_cache_or_reload(). Converts a repo name
    to a directory name according to torch.hub's naming convention.

    Args:
        repo (str): <repo-owner>/<erpo-name>[:tag]

    Returns:
        str: directory name
    """
    from torch.hub import _parse_repo_info
    repo_owner, repo_name, branch = _parse_repo_info(repo)
    normalized_br = branch.replace('/', '_')
    dir_name = '_'.join([repo_owner, repo_name, normalized_br])
    return dir_name


def _uri_to_dir_name(uri: str) -> str:
    """ Determine directory name from a URI. """
    return Path(uri).stem


def get_hubconf_dir_from_cfg(cfg, parent: str = '') -> str:
    """Determine the destination directory path for a module specified
    by an ExternalModuleConfig.

    Args:
        cfg (ExternalModuleConfig): an ExternalModuleConfig
        parent (str, optional): Parent path. Defaults to ''.

    Returns:
        str: directory path
    """
    if cfg.name is not None:
        dir_name = cfg.name
    elif cfg.uri is not None:
        dir_name = __uri_to_dir_name(cfg.uri)
    else:
        dir_name = _repo_name_to_dir_name(cfg.github_repo)

    path = join(parent, dir_name)
    return path


def torch_hub_load_github(repo: str,
                          hubconf_dir: str,
                          tmp_dir: str,
                          entrypoint: str,
                          *args,
                          **kwargs) -> Any:
    """Load an entrypoint from a github repo using torch.hub.load().

    Args:
        repo (str): <repo-owner>/<erpo-name>[:tag]
        entrypoint (str): Name of a callable present in hubconf.py.
        hubconf_dir (str): Where the contents from the uri will finally
        be saved to.
        tmp_dir (str): Where the repo will initially be downloaded.
        *args: Args to be passed to the entrypoint.
        **kwargs: Keyword args to be passed to the entrypoint.

    Returns:
        Any: The output from calling the entrypoint.
    """
    torch.hub.set_dir(tmp_dir)
    out = torch.hub.load(github=repo, model=entrypoint, *args, **kwargs)

    orig_dir = join(tmp_dir, _repo_name_to_dir_name(repo, hub_dir))
    shutil.move(orig_dir, hubconf_dir)

    return out


def torch_hub_load_uri(uri: str, hubconf_dir: str, entrypoint: str,
                       tmp_dir: str, *args, **kwargs) -> Any:
    """Load an entrypoint from:
        - a local uri of a zip file, or
        - a local uri of a directory, or
        - a remote uri of zip file.

    The zip file should either have hubconf.py at the top level or contain
    a single sub-directory that contains hubconf.py at its top level. In the
    latter case, the sub-directory will be copied to hubconf_dir.

    Args:
        uri (str): A URI.
        hubconf_dir (str): The target directory where the contents from the uri
        will finally be saved to.
        entrypoint (str): Name of a callable present in hubconf.py.
        tmp_dir (str): Directory where the zip file will be downloaded to and
        initially extracted.
        *args: Args to be passed to the entrypoint.
        **kwargs: Keyword args to be passed to the entrypoint.

    Returns:
        Any: The output from calling the entrypoint.
    """
    _remove_dir(hubconf_dir)

    filename, ext = splitext(uri)
    is_zip = ext.lower() == '.zip'
    if is_zip:
        # unzip
        zip_path = download_if_needed(uri, tmp_dir)
        unzip_dir = join(tmp_dir, filename)
        _remove_dir(unzip_dir)
        unzip(zip_path, target_dir=unzip_dir)

        # move to hubconf_dir
        unzipped_contents = list(glob(f'{unzip_dir}/*', recursive=False))
        # if the top level only contains a directory
        if (len(unzipped_contents) == 1) and isdir(unzipped_contents[0]):
            sub_dir = unzipped_contents[0]
            shutil.move(sub_dir, hubconf_dir)
        else:
            shutil.move(unzip_dir, hubconf_dir)
        _remove_dir(unzip_dir)
    else:
        # assume uri is local and attempt copying
        shutil.copytree(uri, hubconf_dir)

    out = torch_hub_load_local(hubconf_dir, entrypoint, *args, **kwargs)
    return out


def torch_hub_load_local(hubconf_dir: str, entrypoint: str, *args,
                         **kwargs) -> Any:
    """Adapted from torch.hub.load().

    Args:
        hubconf_dir (str): A directory containing a hubconf.py file.
        entrypoint (str): Name of a callable present in hubconf.py.

    Returns:
        Any: The output from calling the entrypoint.
    """
    from torch.hub import (sys, import_module, MODULE_HUBCONF,
                           _load_entry_from_hubconf)

    verbose = kwargs.get('verbose', True)
    kwargs.pop('verbose', None)

    sys.path.insert(0, hubconf_dir)

    hub_module = import_module(MODULE_HUBCONF, join(hubconf_dir,
                                                    MODULE_HUBCONF))

    entry = _load_entry_from_hubconf(hub_module, entrypoint)
    out = entry(*args, **kwargs)

    sys.path.remove(hubconf_dir)

    return out
