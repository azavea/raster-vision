from typing import TYPE_CHECKING
from os.path import join
import uuid

import numpy as np

from rastervision.pipeline.file_system.utils import make_dir
from rastervision.core.data import SemanticSegmentationLabels
from rastervision.core.data_sample import DataSample
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend)
from rastervision.pytorch_backend.utils import chip_collate_fn_ss
from rastervision.pytorch_learner.utils import predict_scene_ss

if TYPE_CHECKING:
    from rastervision.core.data import DatasetConfig, Scene
    from rastervision.core.rv_pipeline import (
        ChipOptions, SemanticSegmentationPredictOptions)
    from rastervision.pytorch_learner import SemanticSegmentationGeoDataConfig


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

    def predict_scene(self, scene: 'Scene',
                      predict_options: 'SemanticSegmentationPredictOptions'
                      ) -> 'SemanticSegmentationLabels':
        if self.learner is None:
            self.load_model()
        labels = predict_scene_ss(self.learner, scene, predict_options)
        return labels

    def _make_chip_data_config(self, dataset: 'DatasetConfig',
                               chip_options: 'ChipOptions'
                               ) -> 'SemanticSegmentationGeoDataConfig':
        from rastervision.pytorch_learner import (
            SemanticSegmentationGeoDataConfig)
        data_config = SemanticSegmentationGeoDataConfig(
            scene_dataset=dataset, sampling=chip_options.sampling)
        return data_config
