from typing import Dict, Callable, Iterable
from os.path import join
import tempfile
import gc
from itertools import zip_longest

import click
import torch.nn.functional as F

from rastervision.pipeline.file_system import (make_dir, upload_or_copy,
                                               zipdir)
from rastervision.core.backend import Backend, SampleWriter
from rastervision.core.data_sample import DataSample
from rastervision.core.data import (ClassConfig, SceneConfig, Labels,
                                    DatasetConfig)
from rastervision.core.rv_pipeline import RVPipelineConfig, PredictOptions
from rastervision.pytorch_learner import (Learner, LearnerConfig,
                                          GeoDataWindowConfig, GeoDataConfig,
                                          get_base_datasets)


class PyTorchLearnerSampleWriter(SampleWriter):
    def __init__(self, output_uri: str, class_config: ClassConfig,
                 tmp_dir: str):
        """Constructor.

        Args:
            output_uri: URI of directory where zip file of chips should be placed
            class_config: used to convert class ids to names which may be needed for some
                training data formats
            tmp_dir: local directory which is root of any temporary directories that
                are created
        """
        self.output_uri = output_uri
        self.class_config = class_config
        self.tmp_dir = tmp_dir

    def __enter__(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory(dir=self.tmp_dir)
        self.sample_dir = join(self.tmp_dir_obj.name, 'samples')
        make_dir(self.sample_dir)
        self.sample_ind = 0

        return self

    def __exit__(self, type, value, traceback):
        """
        This writes a zip file for a group of scenes at {output_uri}/{uuid}.zip.

        This method is called once per instance of the chip command.
        A number of instances of the chip command can run simultaneously to
        process chips in parallel. The uuid in the zip path above is what allows
        separate instances to avoid overwriting each others' output.
        """
        output_path = join(self.tmp_dir_obj.name, 'output.zip')
        zipdir(self.sample_dir, output_path)
        upload_or_copy(output_path, self.output_uri)
        self.tmp_dir_obj.cleanup()

    def write_sample(self, sample: DataSample):
        """Write a single sample to disk."""
        raise NotImplementedError()


def chunk(iterable: Iterable, n: int) -> Iterable:
    """Collect data into fixed-length chunks or blocks
    Adapted from: https://docs.python.org/3/library/itertools.html
    """
    args = [iter(iterable)] * n
    return zip_longest(*args)


class PyTorchLearnerBackend(Backend):
    """Backend that uses the rastervision.pytorch_learner package to train models."""

    def __init__(self, pipeline_cfg: RVPipelineConfig,
                 learner_cfg: LearnerConfig, tmp_dir: str):
        self.pipeline_cfg = pipeline_cfg
        self.learner_cfg = learner_cfg
        self.tmp_dir = tmp_dir
        self.learner = None

    def train(self, source_bundle_uri=None):
        if source_bundle_uri is not None:
            learner = self._build_learner_from_bundle(
                bundle_uri=source_bundle_uri,
                cfg=self.learner_cfg,
                training=True)
        else:
            learner = self.learner_cfg.build(self.tmp_dir, training=True)
        learner.main()

    def load_model(self):
        self.learner = self._build_learner_from_bundle(training=False)

    def _build_learner_from_bundle(self,
                                   bundle_uri=None,
                                   cfg=None,
                                   training=False):
        if bundle_uri is None:
            bundle_uri = self.learner_cfg.get_model_bundle_uri()
        return Learner.from_model_bundle(
            bundle_uri, self.tmp_dir, cfg=cfg, training=training)

    def get_sample_writer(self):
        raise NotImplementedError()

    def predict(self,
                predict_options: PredictOptions,
                scene: SceneConfig,
                hooks: Dict[str, Callable] = {}) -> Labels:
        if self.learner is None:
            self.load_model()

        dataset = DatasetConfig(
            class_config=self.pipeline_cfg.dataset.class_config,
            train_scenes=[],
            validation_scenes=[],
            test_scenes=[scene])
        window_opts = GeoDataWindowConfig(
            method='sliding',
            size=predict_options.chip_sz,
            stride=predict_options.stride)

        self.learner.cfg.data = self._make_pred_data_config(
            predict_options, dataset, window_opts, scene)
        self.learner.setup_data(
            training=False, overrides={'batch_sz': predict_options.batch_sz})

        labels = self._get_predictions(predict_options, hooks)

        # GeoDataset leaves scenes activated, meaning some tmp dirs might not
        # be cleaned up. This can cause an error later if the root tmp_dir gets
        # garbage collected before these tmp dirs do. So we pre-emptively gc
        # them.
        self.learner.test_dl = None
        self.learner.test_ds = None
        gc.collect()

        return labels

    def _make_pred_data_config(
            self,
            predict_options: PredictOptions,
            scene_dataset: DatasetConfig,
            window_opts: GeoDataWindowConfig,
            scene: SceneConfig,
            hooks: Dict[str, Callable] = {}) -> GeoDataConfig:
        raise NotImplementedError()

    def _get_predictions(self,
                         predict_options: PredictOptions,
                         hooks: Dict[str, Callable] = {}) -> Labels:
        ds = get_base_datasets(self.learner.test_ds)[0]
        dl = self.learner.test_dl

        labels = ds.scene.label_store.empty_labels()

        out_size = predict_options.chip_sz

        windows = chunk(ds.windows, n=predict_options.batch_sz)
        preds = self.learner.iter_predictions(dl)
        it = zip(windows, preds)
        with click.progressbar(it, length=len(dl), label='Predicting') as bar:
            for ws, (xs, _, outs) in bar:
                outs = self.learner.output_to_numpy(outs)
                for w, out in zip(ws, outs):
                    labels[w] = out
                xs = F.interpolate(
                    xs, size=out_size, mode='bilinear', align_corners=False)
                xs = xs.numpy().transpose(0, 2, 3, 1)
                labels = hooks['post-batch'](ws, xs, labels)

        return labels
