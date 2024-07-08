from typing import TYPE_CHECKING
from os.path import join, splitext
import tempfile

import numpy as np
from tqdm.auto import tqdm

from rastervision.pipeline import rv_config_ as rv_config
from rastervision.pipeline.file_system import (make_dir, upload_or_copy,
                                               zipdir)
from rastervision.core.backend import Backend, SampleWriter
from rastervision.core.data.utils.misc import save_img
from rastervision.core.data_sample import DataSample
from rastervision.pytorch_learner.learner import Learner

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from rastervision.core.data import ClassConfig, DatasetConfig, Scene
    from rastervision.core.rv_pipeline import RVPipelineConfig, ChipOptions
    from rastervision.pytorch_learner import DataConfig, LearnerConfig

SPLITS = ['train', 'valid', 'test']


def write_chip(chip: np.ndarray, path: str) -> None:
    """Save chip as either a PNG image or a numpy array."""
    ext = splitext(path)[-1]
    if ext == '.npy':
        np.save(path, chip)
    else:
        chip = chip.astype(np.uint8)
        save_img(chip, path)


def get_image_ext(chip: np.ndarray) -> str:
    """Decide which format to store the image in."""
    if chip.ndim == 2 or chip.shape[-1] == 3:
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

    def get_image_path(self, sample: 'DataSample') -> str:
        """Decide the save location of the image. Also, ensure that the target
        directory exists."""
        split = '' if sample.split is None else sample.split
        img_dir = join(self.sample_dir, split, 'img')
        make_dir(img_dir)

        if sample.scene_id is not None:
            sample_name = f'{sample.scene_id}-{self.sample_ind}'
        else:
            sample_name = f'{self.sample_ind}'
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

    def load_model(self, uri: str | None = None):
        self.learner = self._build_learner_from_bundle(
            bundle_uri=uri, training=False)

    def _build_learner_from_bundle(self,
                                   bundle_uri: str | None = None,
                                   cfg: 'LearnerConfig | None' = None,
                                   training: bool = False):
        if bundle_uri is None:
            bundle_uri = self.learner_cfg.get_model_bundle_uri()
        return Learner.from_model_bundle(
            bundle_uri, self.tmp_dir, cfg=cfg, training=training)

    def get_sample_writer(self):
        raise NotImplementedError()

    def chip_dataset(self,
                     dataset: 'DatasetConfig',
                     chip_options: 'ChipOptions',
                     dataloader_kw: dict = {}) -> None:
        data_config = self._make_chip_data_config(dataset, chip_options)
        train_ds, valid_ds, test_ds = data_config.build(for_chipping=True)

        with self.get_sample_writer() as sample_writer:
            for split, ds in zip(SPLITS, [train_ds, valid_ds, test_ds]):
                if len(ds) == 0:
                    continue
                self.chip_pytorch_dataset(
                    ds,
                    sample_writer=sample_writer,
                    chip_options=chip_options,
                    split=split,
                    dataloader_kw=dataloader_kw)

    def chip_pytorch_dataset(
            self,
            dataset: 'Dataset',
            sample_writer: 'PyTorchLearnerSampleWriter',
            chip_options: 'ChipOptions',
            split: str | None = None,
            dataloader_kw: dict = {},
    ) -> None:
        from torch.utils.data import DataLoader

        num_workers = rv_config.get_namespace_option(
            'rastervision',
            'CHIP_NUM_WORKERS',
            default=self.learner_cfg.data.num_workers)
        batch_size = rv_config.get_namespace_option(
            'rastervision',
            'CHIP_BATCH_SIZE',
            default=self.learner_cfg.solver.batch_sz)

        dl_kw = dict(
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            shuffle=False,
            pin_memory=True)
        dl_kw.update(dataloader_kw)
        dl = DataLoader(dataset, **dl_kw)

        if split is not None:
            desc = f'Chipping {split} scenes.'
        else:
            desc = f'Chipping dataset.'
        with tqdm(total=len(dataset), desc=desc) as bar:
            for (xs, ys), ws in dl:
                for x, y, w in zip(xs, ys, ws):
                    if not chip_options.keep_chip(x, y):
                        continue
                    sample = DataSample(chip=x, label=y, window=w, split=split)
                    sample_writer.write_sample(sample)
                    bar.update(1)

    def predict_scene(self,
                      scene: 'Scene',
                      chip_sz: int,
                      stride: int | None = None):
        raise NotImplementedError()

    def _make_chip_data_config(self, dataset: 'DatasetConfig',
                               chip_options: 'ChipOptions') -> 'DataConfig':
        raise NotImplementedError()
