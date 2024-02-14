from typing import TYPE_CHECKING
from os.path import join
import uuid

from rastervision.pipeline.file_system import make_dir
from rastervision.core.data_sample import DataSample
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend)
from rastervision.pytorch_backend.utils import chip_collate_fn_cc
from rastervision.core.data import ChipClassificationLabels
from rastervision.pytorch_learner.utils import predict_scene_cc

if TYPE_CHECKING:
    from rastervision.core.data import DatasetConfig, Scene
    from rastervision.core.rv_pipeline import ChipOptions, PredictOptions
    from rastervision.pytorch_learner import ClassificationGeoDataConfig


class PyTorchChipClassificationSampleWriter(PyTorchLearnerSampleWriter):
    def write_sample(self, sample: 'DataSample'):
        """
        This writes a training or validation sample to
        (train|valid)/{class_name}/{scene_id}-{ind}.png
        """
        img_path = self.get_image_path(sample)
        self.write_chip(sample.chip, img_path)

        self.sample_ind += 1

    def get_image_path(self, sample: 'DataSample') -> str:
        split = '' if sample.split is None else sample.split
        class_id = sample.label
        class_name = self.class_config.get_name(class_id)
        img_dir = join(self.sample_dir, split, class_name)
        make_dir(img_dir)

        if sample.scene_id is not None:
            sample_name = f'{sample.scene_id}-{self.sample_ind}'
        else:
            sample_name = f'{self.sample_ind}'
        ext = self.get_image_ext(sample.chip)
        img_path = join(img_dir, f'{sample_name}.{ext}')
        return img_path


class PyTorchChipClassification(PyTorchLearnerBackend):
    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, f'{uuid.uuid4()}.zip')
        return PyTorchChipClassificationSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

    def chip_dataset(self,
                     dataset: 'DatasetConfig',
                     chip_options: 'ChipOptions',
                     dataloader_kw: dict = {}) -> None:
        dataloader_kw = dict(**dataloader_kw, collate_fn=chip_collate_fn_cc)
        return super().chip_dataset(dataset, chip_options, dataloader_kw)

    def predict_scene(self, scene: 'Scene', predict_options: 'PredictOptions'
                      ) -> 'ChipClassificationLabels':
        if self.learner is None:
            self.load_model()
        labels = predict_scene_cc(self.learner, scene, predict_options)
        return labels

    def _make_chip_data_config(
            self, dataset: 'DatasetConfig',
            chip_options: 'ChipOptions') -> 'ClassificationGeoDataConfig':
        from rastervision.pytorch_learner import (ClassificationGeoDataConfig)
        data_config = ClassificationGeoDataConfig(
            scene_dataset=dataset, sampling=chip_options.sampling)
        return data_config
