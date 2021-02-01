from typing import Any, Optional
from pathlib import Path
from os.path import join, isdir
import shutil
from glob import glob

import torch.hub

from rastervision.pipeline.file_system import (download_if_needed, unzip)


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


def get_hubconf_dir_from_cfg(cfg, parent: Optional[str] = '') -> str:
    """Determine the destination directory name for a module specified
    by an ExternalModuleConfig. If a parent path is provided, the dir
    name is appended to it.

    Args:
        cfg (ExternalModuleConfig): an ExternalModuleConfig
        parent (str, optional): Parent path. Defaults to ''.

    Returns:
        str: directory name or path
    """
    if cfg.name is not None:
        dir_name = cfg.name
    elif cfg.uri is not None:
        dir_name = _uri_to_dir_name(cfg.uri)
    else:
        dir_name = _repo_name_to_dir_name(cfg.github_repo)

    path = join(parent, dir_name)
    return path


def torch_hub_load_github(repo: str, hubconf_dir: str, tmp_dir: str,
                          entrypoint: str, *args, **kwargs) -> Any:
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
    out = torch.hub.load(repo, entrypoint, *args, source='github', **kwargs)

    orig_dir = join(tmp_dir, _repo_name_to_dir_name(repo))
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

    uri_path = Path(uri)
    is_zip = uri_path.suffix.lower() == '.zip'
    if is_zip:
        # unzip
        zip_path = download_if_needed(uri, tmp_dir)
        unzip_dir = join(tmp_dir, uri_path.stem)
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
    return torch.hub.load(
        hubconf_dir, entrypoint, *args, source='local', **kwargs)
