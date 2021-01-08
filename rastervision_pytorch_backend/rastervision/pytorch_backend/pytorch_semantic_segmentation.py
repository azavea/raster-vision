from os.path import join
from pathlib import Path
import uuid

import numpy as np

from rastervision.pipeline.file_system import (make_dir)
from rastervision.core.data import ClassConfig
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.utils.misc import save_img
from rastervision.core.data_sample import DataSample
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend)


class PyTorchSemanticSegmentationSampleWriter(PyTorchLearnerSampleWriter):
    def __init__(self,
                 output_uri: str,
                 class_config: ClassConfig,
                 tmp_dir: str,
                 img_format: str = 'png',
                 label_format: str = 'png'):
        """Constructor.

        Args:
            output_uri: URI of directory where zip file of chips should be placed
            class_config: used to convert class ids to names which may be needed for some
                training data formats
            tmp_dir: local directory which is root of any temporary directories that
                are created
            img_format: file format to store the image in
            label_format: file format to store the labels in
        """
        super().__init__(output_uri, class_config, tmp_dir)
        self.img_format = img_format
        self.label_format = label_format

    def write_sample(self, sample: DataSample):
        """
        This writes a training or validation sample to
        (train|valid)/img/{scene_id}-{ind}.png and
        (train|valid)/labels/{scene_id}-{ind}.png
        """
        split_name = 'train' if sample.is_train else 'valid'

        sample_dir = Path(self.sample_dir)
        img_dir = sample_dir / split_name / 'img'
        label_dir = sample_dir / split_name / 'labels'

        make_dir(img_dir)
        make_dir(label_dir)

        img = sample.chip
        label_arr = sample.labels.get_label_arr(sample.window).astype(np.uint8)

        img_fmt, label_fmt = self.img_format, self.label_format
        sample_name = f'{sample.scene_id}-{self.sample_ind}'

        # write image
        img_filename = f'{sample_name}.{img_fmt}'
        img_path = img_dir / img_filename
        if img_fmt == 'npy':
            np.save(img_path, img)
        else:
            save_img(img, img_path)

        # write labels
        label_filename = f'{sample_name}.{label_fmt}'
        label_path = label_dir / label_filename
        if label_fmt == 'npy':
            np.save(label_path, label_arr)
        else:
            save_img(label_arr, label_path)

        self.sample_ind += 1


class PyTorchSemanticSegmentation(PyTorchLearnerBackend):
    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, f'{uuid.uuid4()}.zip')
        return PyTorchSemanticSegmentationSampleWriter(
            output_uri,
            self.pipeline_cfg.dataset.class_config,
            self.tmp_dir,
            img_format=self.pipeline_cfg.img_format,
            label_format=self.pipeline_cfg.label_format)

    def predict(self, scene, chips, windows) -> SemanticSegmentationLabels:
        if self.learner is None:
            self.load_model()

        raw_out = scene.label_store.smooth_output
        batch_out = self.learner.numpy_predict(chips, raw_out=raw_out)

        labels = scene.label_store.empty_labels()
        for out, window in zip(batch_out, windows):
            labels[window] = out

        return labels
