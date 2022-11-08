from typing import TYPE_CHECKING, Iterator, Optional
from os.path import join
import uuid

import numpy as np

from rastervision.pipeline.file_system.utils import make_dir
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend)
from rastervision.pytorch_learner.dataset import (
    SemanticSegmentationSlidingWindowGeoDataset)
from rastervision.core.data import SemanticSegmentationLabels

if TYPE_CHECKING:
    from rastervision.core.data_sample import DataSample
    from rastervision.core.data import (Scene, SemanticSegmentationLabelStore)


class PyTorchSemanticSegmentationSampleWriter(PyTorchLearnerSampleWriter):
    def write_sample(self, sample: 'DataSample'):
        """
        This writes a training or validation sample to
        (train|valid)/img/{scene_id}-{ind}.png and
        (train|valid)/labels/{scene_id}-{ind}.png
        """
        split_name = 'train' if sample.is_train else 'valid'

        img = sample.chip
        labels: 'SemanticSegmentationLabels' = sample.labels
        label_arr = labels.get_label_arr(sample.window).astype(np.uint8)

        img_path = self.get_image_path(split_name, sample)
        label_path = self.get_label_path(split_name, sample, label_arr)

        self.write_chip(img, img_path)
        self.write_chip(label_arr, label_path)

        self.sample_ind += 1

    def get_label_path(self, split_name: str, sample: 'DataSample',
                       label_arr: np.ndarray) -> str:
        img_dir = join(self.sample_dir, split_name, 'labels')
        make_dir(img_dir)

        sample_name = f'{sample.scene_id}-{self.sample_ind}'
        ext = self.get_image_ext(label_arr)
        label_path = join(img_dir, f'{sample_name}.{ext}')
        return label_path


class PyTorchSemanticSegmentation(PyTorchLearnerBackend):
    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, f'{uuid.uuid4()}.zip')
        return PyTorchSemanticSegmentationSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

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
            extent=label_store.extent,
            num_classes=len(label_store.class_config),
            crop_sz=crop_sz)

        return labels
