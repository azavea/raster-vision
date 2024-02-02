from typing import TYPE_CHECKING, Iterator, Optional
from os.path import join
import uuid

import numpy as np

from rastervision.pipeline.file_system.utils import make_dir
from rastervision.core.data import SemanticSegmentationLabels
from rastervision.core.data_sample import DataSample
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend)
from rastervision.pytorch_backend.utils import chip_collate_fn_ss
from rastervision.pytorch_learner.dataset import (
    SemanticSegmentationSlidingWindowGeoDataset)
from rastervision.pytorch_learner import SemanticSegmentationGeoDataConfig

if TYPE_CHECKING:
    from rastervision.core.data import (DatasetConfig, Scene,
                                        SemanticSegmentationLabelStore)
    from rastervision.core.rv_pipeline import ChipOptions


class PyTorchSemanticSegmentationSampleWriter(PyTorchLearnerSampleWriter):
    def write_sample(self, sample: 'DataSample'):
        """Write sample.

        This writes a training or validation sample to
        ``(train|valid)/img/{scene_id}-{ind}.png`` and
        ``(train|valid)/labels/{scene_id}-{ind}.png``
        """
        img = sample.chip
        img_path = self.get_image_path(sample)
        self.write_chip(img, img_path)

        if sample.label is not None:
            label_arr: np.ndarray = sample.label
            label_path = self.get_label_path(sample, label_arr)
            self.write_chip(label_arr, label_path)

        self.sample_ind += 1

    def get_label_path(self, sample: 'DataSample',
                       label_arr: np.ndarray) -> str:
        split = '' if sample.split is None else sample.split
        img_dir = join(self.sample_dir, split, 'labels')
        make_dir(img_dir)

        if sample.scene_id is not None:
            sample_name = f'{sample.scene_id}-{self.sample_ind}'
        else:
            sample_name = f'{self.sample_ind}'
        ext = self.get_image_ext(label_arr)
        label_path = join(img_dir, f'{sample_name}.{ext}')
        return label_path


class PyTorchSemanticSegmentation(PyTorchLearnerBackend):
    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, f'{uuid.uuid4()}.zip')
        return PyTorchSemanticSegmentationSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

    def chip_dataset(self,
                     dataset: 'DatasetConfig',
                     chip_options: 'ChipOptions',
                     dataloader_kw: dict = {}) -> None:
        dataloader_kw = dict(**dataloader_kw, collate_fn=chip_collate_fn_ss)
        return super().chip_dataset(dataset, chip_options, dataloader_kw)

    def predict_scene(
            self,
            scene: 'Scene',
            chip_sz: int,
            stride: Optional[int] = None,
            crop_sz: Optional[int] = None) -> 'SemanticSegmentationLabels':

        if scene.label_store is None:
            raise ValueError(
                f'Scene.label_store is not set for scene {scene.id}')

        if stride is None:
            stride = chip_sz

        if self.learner is None:
            self.load_model()

        label_store: 'SemanticSegmentationLabelStore' = scene.label_store
        raw_out = label_store.smooth_output

        # Important to use self.learner.cfg.data instead of
        # self.learner_cfg.data because of the updates
        # Learner.from_model_bundle() makes to the custom transforms.
        base_tf, _ = self.learner.cfg.data.get_data_transforms()
        pad_direction = 'end' if crop_sz is None else 'both'
        ds = SemanticSegmentationSlidingWindowGeoDataset(
            scene,
            size=chip_sz,
            stride=stride,
            pad_direction=pad_direction,
            transform=base_tf)

        predictions: Iterator[np.ndarray] = self.learner.predict_dataset(
            ds,
            raw_out=raw_out,
            numpy_out=True,
            predict_kw=dict(out_shape=(chip_sz, chip_sz)),
            progress_bar=True,
            progress_bar_kw=dict(desc=f'Making predictions on {scene.id}'))

        labels = SemanticSegmentationLabels.from_predictions(
            ds.windows,
            predictions,
            smooth=raw_out,
            extent=scene.extent,
            num_classes=len(label_store.class_config),
            crop_sz=crop_sz)

        return labels

    def _make_chip_data_config(
            self, dataset: 'DatasetConfig',
            chip_options: 'ChipOptions') -> SemanticSegmentationGeoDataConfig:
        data_config = SemanticSegmentationGeoDataConfig(
            scene_dataset=dataset, sampling=chip_options.sampling)
        return data_config
