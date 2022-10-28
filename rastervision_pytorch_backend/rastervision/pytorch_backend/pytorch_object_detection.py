from typing import TYPE_CHECKING, Dict, Iterator, Optional
from os.path import join, basename
import uuid

from rastervision.pipeline.file_system import json_to_file
from rastervision.core.data.label import ObjectDetectionLabels
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend)
from rastervision.pytorch_learner.dataset import (
    ObjectDetectionSlidingWindowGeoDataset)

if TYPE_CHECKING:
    import numpy as np
    from rastervision.core.data import Scene
    from rastervision.core.data_sample import DataSample


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
            if len(self.splits[split]['images']) > 0:
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
        split_name = 'train' if sample.is_train else 'valid'

        img_path = self.get_image_path(split_name, sample)
        self.write_chip(sample.chip, img_path)
        self.update_coco_data(split_name, sample, img_path)
        self.sample_ind += 1

    def update_coco_data(self, split_name: str, sample: 'DataSample',
                         img_path: str):
        images = self.splits[split_name]['images']
        annotations = self.splits[split_name]['annotations']

        images.append({
            'file_name': basename(img_path),
            'id': self.sample_ind,
            'height': sample.chip.shape[0],
            'width': sample.chip.shape[1]
        })

        npboxes = sample.labels.get_npboxes()
        npboxes = ObjectDetectionLabels.global_to_local(npboxes, sample.window)
        for box_ind, (box, class_id) in enumerate(
                zip(npboxes, sample.labels.get_class_ids())):
            bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
            bbox = [int(i) for i in bbox]
            annotations.append({
                'id': '{}-{}'.format(self.sample_ind, box_ind),
                'image_id': self.sample_ind,
                'bbox': bbox,
                'category_id': int(class_id)
            })


class PyTorchObjectDetection(PyTorchLearnerBackend):
    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, f'{uuid.uuid4()}.zip')
        return PyTorchObjectDetectionSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

    def predict_scene(self,
                      scene: 'Scene',
                      chip_sz: int,
                      stride: Optional[int] = None) -> ObjectDetectionLabels:
        if stride is None:
            stride = chip_sz

        if self.learner is None:
            self.load_model()

        # Important to use self.learner.cfg.data instead of
        # self.learner_cfg.data because of the updates
        # Learner.from_model_bundle() makes to the custom transforms.
        base_tf, _ = self.learner.cfg.data.get_data_transforms()
        ds = ObjectDetectionSlidingWindowGeoDataset(
            scene, size=chip_sz, stride=stride, transform=base_tf)

        predictions: Iterator[Dict[str, 'np.ndarray']] = (
            self.learner.predict_dataset(
                ds,
                raw_out=True,
                numpy_out=True,
                progress_bar=True,
                progress_bar_kw=dict(desc=f'Making predictions on {scene.id}'))
        )

        labels = ObjectDetectionLabels.from_predictions(
            ds.windows, predictions)

        return labels
