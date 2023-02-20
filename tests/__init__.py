from typing import Callable
import os


def data_file_path(rel_path: str) -> str:
    data_dir = os.path.join(os.path.dirname(__file__), 'data_files')
    return os.path.join(data_dir, rel_path)


def test_config_upgrader(cfg_class: type, old_cfg_dict: dict,
                         upgrader: Callable, curr_version: int) -> None:
    """Try to use upgrader to update cfg dict to curr_version."""
    from rastervision.pipeline.config import build_config

    for i in range(curr_version):
        old_cfg_dict = upgrader(old_cfg_dict, version=i)
    new_cfg = build_config(old_cfg_dict)
    assert isinstance(new_cfg, cfg_class)
