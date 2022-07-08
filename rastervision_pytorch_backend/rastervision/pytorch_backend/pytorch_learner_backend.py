from typing import TYPE_CHECKING, Optional
from os.path import join, splitext
import tempfile

import numpy as np

from rastervision.pipeline.file_system import (make_dir, upload_or_copy,
                                               zipdir)
from rastervision.core.backend import Backend, SampleWriter
from rastervision.core.utils.misc import save_img
from rastervision.pytorch_learner.learner import Learner

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig, Scene
    from rastervision.core.data_sample import DataSample
    from rastervision.core.rv_pipeline import RVPipelineConfig
    from rastervision.pytorch_learner.learner_config import LearnerConfig


def write_chip(chip: np.ndarray, path: str) -> None:
    """Save chip as either a PNG image or a numpy array."""
    ext = splitext(path)[-1]
    if ext == '.npy':
        np.save(path, chip)
    else:
        save_img(chip, path)


def get_image_ext(chip: np.ndarray) -> str:
    """Decide which format to store the image in."""
    if len(chip.shape) not in (2, 3):
        raise ValueError('chip shape must be (H, W) or (H, W, C)')
    if len(chip.shape) == 2 or chip.shape[-1] == 3:
        return 'png'
    else:
        return 'npy'


class PyTorchLearnerSampleWriter(SampleWriter):
    def __init__(self, output_uri: str, class_config: 'ClassConfig',
                 tmp_dir: str):
        """Constructor.

        Args:
            output_uri (str): URI of directory where zip file of chips should
                be placed.
            class_config (ClassConfig): used to convert class ids to names
                which may be needed for some training data formats.
            tmp_dir (str): local directory which is root of any temporary
                directories that are created.
        """
        self.output_uri = output_uri
        self.class_config = class_config
        self.tmp_dir = tmp_dir

    def __enter__(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory(dir=self.tmp_dir)
        self.sample_dir = join(self.tmp_dir_obj.name, 'samples')
        make_dir(self.sample_dir)
        self.sample_ind = 0

        return self

    def __exit__(self, type, value, traceback):
        """
        This writes a zip file for a group of scenes at {output_uri}/{uuid}.zip.

        This method is called once per instance of the chip command.
        A number of instances of the chip command can run simultaneously to
        process chips in parallel. The uuid in the zip path above is what allows
        separate instances to avoid overwriting each others' output.
        """
        output_path = join(self.tmp_dir_obj.name, 'output.zip')
        zipdir(self.sample_dir, output_path)
        upload_or_copy(output_path, self.output_uri)
        self.tmp_dir_obj.cleanup()

    def write_sample(self, sample: 'DataSample') -> None:
        """Write a single sample to disk."""
        raise NotImplementedError()

    def get_image_path(self, split_name: str, sample: 'DataSample') -> str:
        """Decide the save location of the image. Also, ensure that the target
        directory exists."""
        img_dir = join(self.sample_dir, split_name, 'img')
        make_dir(img_dir)

        sample_name = f'{sample.scene_id}-{self.sample_ind}'
        ext = self.get_image_ext(sample.chip)
        img_path = join(img_dir, f'{sample_name}.{ext}')
        return img_path

    def get_image_ext(self, chip: np.ndarray) -> str:
        """Decide which format to store the image in."""
        return get_image_ext(chip)

    def write_chip(self, chip: np.ndarray, path: str) -> None:
        """Save chip as either a PNG image or a numpy array."""
        write_chip(chip, path)


class PyTorchLearnerBackend(Backend):
    """Backend that uses the rastervision.pytorch_learner package to train models."""

    def __init__(self, pipeline_cfg: 'RVPipelineConfig',
                 learner_cfg: 'LearnerConfig', tmp_dir: str):
        self.pipeline_cfg = pipeline_cfg
        self.learner_cfg = learner_cfg
        self.tmp_dir = tmp_dir
        self.learner = None

    def train(self, source_bundle_uri=None):
        if source_bundle_uri is not None:
            learner = self._build_learner_from_bundle(
                bundle_uri=source_bundle_uri,
                cfg=self.learner_cfg,
                training=True)
        else:
            learner = self.learner_cfg.build(self.tmp_dir, training=True)
        learner.main()

    def load_model(self):
        self.learner = self._build_learner_from_bundle(training=False)

    def _build_learner_from_bundle(self,
                                   bundle_uri=None,
                                   cfg=None,
                                   training=False):
        if bundle_uri is None:
            bundle_uri = self.learner_cfg.get_model_bundle_uri()
        return Learner.from_model_bundle(
            bundle_uri, self.tmp_dir, cfg=cfg, training=training)

    def get_sample_writer(self):
        raise NotImplementedError()

    def predict_scene(self,
                      scene: 'Scene',
                      chip_sz: int,
                      stride: Optional[int] = None):
        raise NotImplementedError()
