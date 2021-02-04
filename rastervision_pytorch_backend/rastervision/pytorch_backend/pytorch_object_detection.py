from typing import Dict, Callable
from os.path import join
import uuid

import click

from rastervision.pipeline.file_system import (make_dir, json_to_file)
from rastervision.core.data import (ObjectDetectionLabels, SceneConfig,
                                    DatasetConfig)
from rastervision.core.rv_pipeline import (ObjectDetectionPredictOptions)
from rastervision.core.utils.misc import save_img
from rastervision.core.data_sample import DataSample
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend, chunk)
from rastervision.pytorch_learner import (
    ObjectDetectionGeoDataConfig, GeoDataWindowConfig, get_base_datasets)


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

    def write_sample(self, sample: DataSample):
        """
        This writes a training or validation sample to
        (train|valid)/img/{scene_id}-{ind}.png and updates
        some COCO data structures.
        """
        split = 'train' if sample.is_train else 'valid'
        split_dir = join(self.sample_dir, split)
        img_dir = join(split_dir, 'img')
        make_dir(img_dir)
        img_fn = '{}-{}.png'.format(sample.scene_id, self.sample_ind)
        img_path = join(img_dir, img_fn)
        save_img(sample.chip, img_path)

        images = self.splits[split]['images']
        annotations = self.splits[split]['annotations']

        images.append({
            'file_name': img_fn,
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

        self.sample_ind += 1


class PyTorchObjectDetection(PyTorchLearnerBackend):
    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, '{}.zip'.format(
            str(uuid.uuid4())))
        return PyTorchObjectDetectionSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

    def _get_predictions(
            self,
            predict_options: ObjectDetectionPredictOptions,
            hooks: Dict[str, Callable] = {}) -> ObjectDetectionLabels:
        ds = get_base_datasets(self.learner.test_ds)[0]
        dl = self.learner.test_dl

        labels = ds.scene.label_store.empty_labels()

        windows = chunk(ds.windows, n=predict_options.batch_sz)
        preds = self.learner.iter_predictions(dl, raw_out=False)

        it = zip(windows, preds)
        with click.progressbar(it, length=len(dl), label='Predicting') as bar:
            for ws, (xs, _, outs) in bar:
                outs = self.learner.output_to_numpy(outs)
                for w, x, out in zip(ws, xs, outs):
                    out['boxes'] = ObjectDetectionLabels.local_to_global(
                        out['boxes'], w)
                    labels[w] = out

        return labels

    def _make_pred_data_config(
            self,
            predict_options: ObjectDetectionPredictOptions,
            scene_dataset: DatasetConfig,
            window_opts: GeoDataWindowConfig,
            scene: SceneConfig,
            hooks: Dict[str, Callable] = {}) -> ObjectDetectionGeoDataConfig:
        cfg = self.learner.cfg.data
        return ObjectDetectionGeoDataConfig(
            class_names=cfg.class_names,
            class_colors=cfg.class_colors,
            img_sz=cfg.img_sz,
            num_workers=cfg.num_workers,
            base_transform=cfg.base_transform,
            plot_options=cfg.plot_options,
            scene_dataset=scene_dataset,
            window_opts=window_opts)
