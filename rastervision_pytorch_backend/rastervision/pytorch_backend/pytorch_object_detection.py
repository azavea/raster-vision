from typing import TYPE_CHECKING
from os.path import join, basename
import uuid

from rastervision.pipeline.file_system import json_to_file
from rastervision.core.data_sample import DataSample
from rastervision.core.data.label import ObjectDetectionLabels
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend)
from rastervision.pytorch_backend.utils import chip_collate_fn_od
from rastervision.pytorch_learner.utils import predict_scene_od

if TYPE_CHECKING:
    from rastervision.core.data import DatasetConfig, Scene
    from rastervision.core.rv_pipeline import (ChipOptions,
                                               ObjectDetectionPredictOptions)
    from rastervision.pytorch_learner.object_detection_utils import BoxList
    from rastervision.pytorch_learner.object_detection_learner_config import (
        ObjectDetectionGeoDataConfig)


class PyTorchObjectDetectionSampleWriter(PyTorchLearnerSampleWriter):
    """Writes data in COCO format."""

    def __enter__(self):
        super().__enter__()

        self.splits = {
            'train': {
                'images': [],
                'annotations': []
            },
            'valid': {
                'images': [],
                'annotations': []
            }
        }
        self.categories = [{
            'id': class_id,
            'name': class_name
        } for class_id, class_name in enumerate(self.class_config.names)]

        return self

    def __exit__(self, type, value, traceback):
        """This writes label files in COCO format to (train|valid)/labels.json"""
        for split in ['train', 'valid']:
            if len(self.splits[split]['images']) == 0:
                continue
            split_dir = join(self.sample_dir, split)
            labels_path = join(split_dir, 'labels.json')

            images = self.splits[split]['images']
            annotations = self.splits[split]['annotations']
            coco_dict = {
                'images': images,
                'annotations': annotations,
                'categories': self.categories
            }
            json_to_file(coco_dict, labels_path)

        super().__exit__(type, value, traceback)

    def write_sample(self, sample: 'DataSample'):
        """
        This writes a training or validation sample to
        (train|valid)/img/{scene_id}-{ind}.png and updates
        some COCO data structures.
        """
        img_path = self.get_image_path(sample)
        self.write_chip(sample.chip, img_path)
        self.update_coco_data(sample, img_path)
        self.sample_ind += 1

    def update_coco_data(self, sample: 'DataSample', img_path: str):
        split = 'default' if sample.split is None else sample.split
        images = self.splits[split]['images']
        annotations = self.splits[split]['annotations']

        images.append({
            'file_name': basename(img_path),
            'id': self.sample_ind,
            'height': sample.chip.shape[0],
            'width': sample.chip.shape[1]
        })

        boxlist: 'BoxList' = sample.label
        npboxes = boxlist.convert_boxes('xywh')
        class_ids = boxlist.get_field('class_ids')
        for i, (bbox, class_id) in enumerate(zip(npboxes, class_ids)):
            bbox = [int(v) for v in bbox]
            class_id = int(class_id)
            annotations.append({
                'id': f'{self.sample_ind}-{i}',
                'image_id': self.sample_ind,
                'bbox': bbox,
                'category_id': class_id,
            })


class PyTorchObjectDetection(PyTorchLearnerBackend):
    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, f'{uuid.uuid4()}.zip')
        return PyTorchObjectDetectionSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

    def chip_dataset(self,
                     dataset: 'DatasetConfig',
                     chip_options: 'ChipOptions',
                     dataloader_kw: dict = {}) -> None:
        dataloader_kw = dict(**dataloader_kw, collate_fn=chip_collate_fn_od)
        return super().chip_dataset(dataset, chip_options, dataloader_kw)

    def predict_scene(self, scene: 'Scene',
                      predict_options: 'ObjectDetectionPredictOptions'
                      ) -> ObjectDetectionLabels:
        if self.learner is None:
            self.load_model()
        labels = predict_scene_od(self.learner, scene, predict_options)
        return labels

    def _make_chip_data_config(
            self, dataset: 'DatasetConfig',
            chip_options: 'ChipOptions') -> 'ObjectDetectionGeoDataConfig':
        from rastervision.pytorch_learner import (ObjectDetectionGeoDataConfig)
        data_config = ObjectDetectionGeoDataConfig(
            scene_dataset=dataset, sampling=chip_options.sampling)
        return data_config
