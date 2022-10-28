from typing import TYPE_CHECKING, Iterator, Optional
from os.path import join
import uuid

from rastervision.pipeline.file_system import (make_dir)
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend)
from rastervision.pytorch_learner.dataset import (
    ClassificationSlidingWindowGeoDataset)
from rastervision.core.data import ChipClassificationLabels

if TYPE_CHECKING:
    import numpy as np
    from rastervision.core.data import Scene
    from rastervision.core.data_sample import DataSample


class PyTorchChipClassificationSampleWriter(PyTorchLearnerSampleWriter):
    def write_sample(self, sample: 'DataSample'):
        """
        This writes a training or validation sample to
        (train|valid)/{class_name}/{scene_id}-{ind}.png
        """
        class_id = sample.labels.get_cell_class_id(sample.window)
        # If a chip is not associated with a class, don't
        # use it in training data.
        if class_id is None:
            return

        split_name = 'train' if sample.is_train else 'valid'
        img_path = self.get_image_path(split_name, sample, class_id)
        self.write_chip(sample.chip, img_path)

        self.sample_ind += 1

    def get_image_path(self, split_name: str, sample: 'DataSample',
                       class_id: int) -> str:
        class_name = self.class_config.names[class_id]
        img_dir = join(self.sample_dir, split_name, class_name)
        make_dir(img_dir)

        sample_name = f'{sample.scene_id}-{self.sample_ind}'
        ext = self.get_image_ext(sample.chip)
        img_path = join(img_dir, f'{sample_name}.{ext}')
        return img_path


class PyTorchChipClassification(PyTorchLearnerBackend):
    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, f'{uuid.uuid4()}.zip')
        return PyTorchChipClassificationSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

    def predict_scene(self,
                      scene: 'Scene',
                      chip_sz: int,
                      stride: Optional[int] = None
                      ) -> 'ChipClassificationLabels':
        if stride is None:
            stride = chip_sz

        if self.learner is None:
            self.load_model()

        # Important to use self.learner.cfg.data instead of
        # self.learner_cfg.data because of the updates
        # Learner.from_model_bundle() makes to the custom transforms.
        base_tf, _ = self.learner.cfg.data.get_data_transforms()
        ds = ClassificationSlidingWindowGeoDataset(
            scene, size=chip_sz, stride=stride, transform=base_tf)

        predictions: Iterator['np.array'] = self.learner.predict_dataset(
            ds,
            raw_out=True,
            numpy_out=True,
            progress_bar=True,
            progress_bar_kw=dict(desc=f'Making predictions on {scene.id}'))

        labels = ChipClassificationLabels.from_predictions(
            ds.windows, predictions)

        return labels
