from typing import Dict, Callable
from os.path import join
from pathlib import Path
import uuid

import numpy as np
import torch.nn.functional as F
import click

from rastervision.pipeline.file_system import (make_dir)
from rastervision.core.data import ClassConfig, SceneConfig, DatasetConfig
from rastervision.core.rv_pipeline import PredictOptions
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.utils.misc import save_img
from rastervision.core.data_sample import DataSample
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend, chunk)
from rastervision.pytorch_learner import (GeoDataWindowConfig,
                                          SemanticSegmentationGeoDataConfig)


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

    def _get_predictions(self,
                         predict_options: PredictOptions,
                         hooks: Dict[str, Callable] = {}
                         ) -> SemanticSegmentationLabels:
        ds = self.learner.test_ds.datasets[0]  # index into the ConcatDataset
        dl = self.learner.test_dl

        raw_out = ds.scene.label_store.smooth_output
        labels = ds.scene.label_store.empty_labels()

        out_size = predict_options.chip_sz

        windows = chunk(ds.windows, n=predict_options.batch_sz)
        preds = self.learner.iter_predictions(dl, raw_out=True)
        it = zip(windows, preds)
        with click.progressbar(it, length=len(dl), label='Predicting') as bar:
            for ws, (xs, _, outs) in bar:
                # resize
                xs = F.interpolate(
                    xs, size=out_size, mode='bilinear', align_corners=False)
                outs = F.interpolate(
                    outs, size=out_size, mode='bilinear', align_corners=False)

                if not raw_out:
                    outs = self.learner.prob_to_pred(outs)

                outs = self.learner.output_to_numpy(outs)

                for w, x, out in zip(ws, xs, outs):
                    labels[w] = out
                xs = xs.numpy().transpose(0, 2, 3, 1)
                labels = hooks['post-batch'](ws, xs, labels)

        return labels

    def _make_pred_data_config(self,
                               predict_options: PredictOptions,
                               scene_dataset: DatasetConfig,
                               window_opts: GeoDataWindowConfig,
                               scene: SceneConfig,
                               hooks: Dict[str, Callable] = {}
                               ) -> SemanticSegmentationGeoDataConfig:
        cfg = self.learner.cfg.data
        return SemanticSegmentationGeoDataConfig(
            class_names=cfg.class_names,
            class_colors=cfg.class_colors,
            img_sz=cfg.img_sz,
            num_workers=cfg.num_workers,
            channel_display_groups=cfg.channel_display_groups,
            img_channels=cfg.img_channels,
            base_transform=cfg.base_transform,
            plot_options=cfg.plot_options,
            scene_dataset=scene_dataset,
            window_opts=window_opts)
