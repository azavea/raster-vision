from os.path import join
import uuid

import numpy as np

from rastervision.pipeline.file_system import (make_dir)
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.utils.misc import save_img
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
        label_arr = sample.labels.get_label_arr(sample.window).astype(np.uint8)

        img_dir = join(self.sample_dir, split_name, 'img')
        labels_dir = join(self.sample_dir, split_name, 'labels')
        make_dir(img_dir)
        make_dir(labels_dir)

        img_path = join(img_dir, '{}-{}.png'.format(sample.scene_id,
                                                    self.sample_ind))
        labels_path = join(
            labels_dir, '{}-{}.png'.format(sample.scene_id, self.sample_ind))
        save_img(sample.chip, img_path)
        save_img(label_arr, labels_path)

        self.sample_ind += 1


class PyTorchSemanticSegmentation(PyTorchLearnerBackend):
    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, '{}.zip'.format(
            str(uuid.uuid4())))
        return PyTorchSemanticSegmentationSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

    def predict(self, chips, windows):
        if self.learner is None:
            self.load_model()

        batch_out = self.learner.numpy_predict(chips, raw_out=False)
        labels = SemanticSegmentationLabels()
        for out, window in zip(batch_out, windows):
            labels.set_label_arr(window, out)

        return labels
