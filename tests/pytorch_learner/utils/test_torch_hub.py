from typing import Callable
import unittest
from tempfile import TemporaryDirectory
from os.path import join, exists
from os import makedirs

from torch import nn

from rastervision.pytorch_learner.utils.torch_hub import (
    _remove_dir, _repo_name_to_dir_name, _uri_to_dir_name,
    get_hubconf_dir_from_cfg, torch_hub_load_github, torch_hub_load_local,
    torch_hub_load_uri)
from rastervision.pytorch_learner import ExternalModuleConfig


class TestTorchHubUtils(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_remove_dir(self):
        with TemporaryDirectory(dir='/opt/data/tmp') as tmp_dir:
            dir_to_be_removed = join(tmp_dir, 'dir_to_be_removed')
            makedirs(dir_to_be_removed, exist_ok=True)
            _remove_dir(dir_to_be_removed)
            self.assertFalse(exists(dir_to_be_removed))

    def test_repo_name_to_dir_name(self):
        repo_name = 'AdeelH/pytorch-fpn:0.3'
        dir_name = _repo_name_to_dir_name(repo_name)
        self.assertEqual(dir_name, 'AdeelH_pytorch-fpn_0.3')

    def test_uri_to_dir_name(self):
        uri = 's3://some/path/to/repo.zip'
        dir_name = _uri_to_dir_name(uri)
        self.assertEqual(dir_name, 'repo')

    def test_get_hubconf_dir_from_cfg(self):
        # config using a github repo, with a name
        cfg_repo_w_name = ExternalModuleConfig(
            github_repo='AdeelH/pytorch-fpn:0.3',
            name='fpn',
            entrypoint='make_fpn_resnet')
        dir_name = get_hubconf_dir_from_cfg(cfg_repo_w_name)
        self.assertEqual(dir_name, 'fpn')

        # config using a uri, without a name
        cfg_repo_wo_name = ExternalModuleConfig(
            github_repo='AdeelH/pytorch-fpn:0.3', entrypoint='make_fpn_resnet')
        dir_name = get_hubconf_dir_from_cfg(cfg_repo_wo_name)
        self.assertEqual(dir_name, 'AdeelH_pytorch-fpn_0.3')

        # config using a github repo, with a name
        cfg_uri_w_name = ExternalModuleConfig(
            uri='s3://some/path/to/repo.zip',
            name='fpn',
            entrypoint='make_fpn_resnet')
        dir_name = get_hubconf_dir_from_cfg(cfg_uri_w_name)
        self.assertEqual(dir_name, 'fpn')

        # config using a uri, without a name
        cfg_uri_wo_name = ExternalModuleConfig(
            uri='s3://some/path/to/repo.zip', entrypoint='make_fpn_resnet')
        dir_name = get_hubconf_dir_from_cfg(cfg_uri_wo_name)
        self.assertEqual(dir_name, 'repo')

    def test_torch_hub_load(self):
        with TemporaryDirectory(dir='/opt/data/tmp') as tmp_dir:
            # github
            hubconf_dir = join(tmp_dir, 'focal_loss')
            loss = torch_hub_load_github(
                repo='AdeelH/pytorch-multi-class-focal-loss:1.1',
                hubconf_dir=hubconf_dir,
                entrypoint='focal_loss',
                alpha=[.75, .25],
                gamma=2)
            self.assertIsInstance(loss, nn.Module)
            self.assertEqual(loss.alpha.tolist(), [.75, .25])
            self.assertEqual(loss.gamma, 2)
            del loss

            # local, via torch_hub_load_local
            loss = torch_hub_load_local(
                hubconf_dir=hubconf_dir,
                entrypoint='focal_loss',
                alpha=[.75, .25],
                gamma=2)
            self.assertIsInstance(loss, nn.Module)
            self.assertEqual(loss.alpha.tolist(), [.75, .25])
            self.assertEqual(loss.gamma, 2)
            del loss

            # local, via torch_hub_load_uri
            loss = torch_hub_load_uri(
                uri=hubconf_dir,
                hubconf_dir=hubconf_dir,
                entrypoint='focal_loss',
                alpha=[.75, .25],
                gamma=2)
            self.assertIsInstance(loss, nn.Module)
            self.assertEqual(loss.alpha.tolist(), [.75, .25])
            self.assertEqual(loss.gamma, 2)
            del loss

        # torch_hub_load_uri, zip
        with TemporaryDirectory(dir='/opt/data/tmp') as tmp_dir:
            hubconf_dir = join(tmp_dir, 'focal_loss')
            loss = torch_hub_load_uri(
                uri=
                'https://github.com/AdeelH/pytorch-multi-class-focal-loss/archive/refs/tags/1.1.zip',  # noqa
                hubconf_dir=hubconf_dir,
                entrypoint='focal_loss',
                alpha=[.75, .25],
                gamma=2)
            self.assertIsInstance(loss, nn.Module)
            self.assertEqual(loss.alpha.tolist(), [.75, .25])
            self.assertEqual(loss.gamma, 2)


if __name__ == '__main__':
    unittest.main()
