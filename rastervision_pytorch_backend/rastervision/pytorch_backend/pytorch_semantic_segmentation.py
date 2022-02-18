from os.path import join
import uuid

import numpy as np

from rastervision.pipeline.file_system import (make_dir)
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data_sample import DataSample
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend)


class PyTorchSemanticSegmentationSampleWriter(PyTorchLearnerSampleWriter):
    def write_sample(self, sample: DataSample):
        """
        This writes a training or validation sample to
        (train|valid)/img/{scene_id}-{ind}.png and
        (train|valid)/labels/{scene_id}-{ind}.png
        """
        split_name = 'train' if sample.is_train else 'valid'

        img = sample.chip
        labels: SemanticSegmentationLabels = sample.labels
        label_arr = labels.get_label_arr(sample.window).astype(np.uint8)

        img_path = self.get_image_path(split_name, sample)
        label_path = self.get_label_path(split_name, sample, label_arr)

        self.write_chip(img, img_path)
        self.write_chip(label_arr, label_path)

        self.sample_ind += 1

    def get_label_path(self, split_name: str, sample: DataSample,
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

    def predict(self, scene, chips, windows) -> SemanticSegmentationLabels:
        if self.learner is None:
            self.load_model()

        raw_out = scene.label_store.smooth_output
        batch_out = self.learner.numpy_predict(chips, raw_out=raw_out)

        labels = scene.label_store.empty_labels()
        for out, window in zip(batch_out, windows):
            labels[window] = out

        return labels
