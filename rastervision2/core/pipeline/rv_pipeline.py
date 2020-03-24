import logging
from os.path import join
import tempfile

import numpy as np

from rastervision2.pipeline.pipeline import Pipeline
from rastervision2.core import DataSample
from rastervision2.core.pipeline import TRAIN, VALIDATION
from rastervision2.pipeline.file_system.utils import (
    download_or_copy, zipdir, get_local_path, upload_or_copy)

log = logging.getLogger(__name__)


class RVPipeline(Pipeline):
    commands = ['analyze', 'chip', 'train', 'predict', 'eval', 'bundle']
    split_commands = ['chip', 'predict']
    gpu_commands = ['train', 'predict']

    def __init__(self, config, tmp_dir):
        super().__init__(config, tmp_dir)

        self.backend = None

    def analyze(self):
        class_config = self.config.dataset.class_config
        scenes = [
            s.build(class_config, self.tmp_dir, use_transformers=False)
            for s in self.config.dataset.train_scenes
        ]
        analyzers = [a.build() for a in self.config.analyzers]
        for analyzer in analyzers:
            log.info('Running analyzers: {}...'.format(
                type(analyzer).__name__))
            analyzer.process(scenes, self.tmp_dir)

    def get_train_windows(self, scene):
        """Return the training windows for a Scene.

        The training windows represent the spatial extent of the training
        chips to generate.

        Args:
            scene: Scene to generate windows for

        Returns:
            list of Boxes
        """
        raise NotImplementedError()

    def get_train_labels(self, window, scene):
        """Return the training labels in a window for a scene.

        Args:
            window: Box
            scene: Scene

        Returns:
            Labels that lie within window
        """
        raise NotImplementedError()

    def process_sample(self, sample):
        return sample

    def chip(self, split_ind=0, num_splits=1):
        cfg = self.config
        backend = cfg.backend.build(cfg, self.tmp_dir)
        dataset = cfg.dataset.get_split_config(split_ind, num_splits)
        if not dataset.train_scenes and not dataset.validation_scenes:
            return

        class_cfg = dataset.class_config
        with backend.get_sample_writer() as writer:

            def chip_scene(scene, split):
                with scene.activate():
                    log.info('Making {} chips for scene: {}'.format(
                        split, scene.id))
                    windows = self.get_train_windows(scene)
                    for window in windows:
                        chip = scene.raster_source.get_chip(window)
                        labels = self.get_train_labels(window, scene)
                        sample = DataSample(
                            chip=chip,
                            window=window,
                            labels=labels,
                            scene_id=str(scene.id),
                            is_train=split == TRAIN)
                        sample = self.process_sample(sample)
                        writer.write_sample(sample)

            for s in dataset.train_scenes:
                chip_scene(s.build(class_cfg, self.tmp_dir), TRAIN)
            for s in dataset.validation_scenes:
                chip_scene(s.build(class_cfg, self.tmp_dir), VALIDATION)

    def train(self):
        backend = self.config.backend.build(self.config, self.tmp_dir)
        backend.train()

    def post_process_batch(self, windows, chips, labels):
        return labels

    def post_process_predictions(self, labels, scene):
        """Runs a post-processing step on labels at end of prediction.

        Returns:
            Labels
        """
        return labels

    def get_predict_windows(self, extent):
        """Return windows to compute predictions for.

        Args:
            extent: Box representing extent of RasterSource

        Returns:
            list of Boxes
        """
        chip_sz = stride = self.config.predict_chip_sz
        return extent.get_windows(chip_sz, stride)

    def predict(self, split_ind=0, num_splits=1):
        # Cache backend so subsquent calls will be faster. This is useful for
        # the predictor.
        if self.backend is None:
            self.backend = self.config.backend.build(self.config, self.tmp_dir)
            self.backend.load_model()

        class_config = self.config.dataset.class_config
        dataset = self.config.dataset.get_split_config(split_ind, num_splits)

        def _predict(scenes):
            for scene in scenes:
                with scene.activate():
                    labels = self.predict_scene(scene, self.backend)
                    label_store = scene.prediction_label_store
                    label_store.save(labels)

        _predict([
            s.build(class_config, self.tmp_dir)
            for s in dataset.validation_scenes
        ])
        if dataset.test_scenes:
            _predict([
                s.build(class_config, self.tmp_dir)
                for s in dataset.test_scenes
            ])

    def predict_scene(self, scene, backend):
        """Predict on a single scene, and return the labels."""
        log.info('Making predictions for scene')
        raster_source = scene.raster_source
        label_store = scene.prediction_label_store
        labels = label_store.empty_labels()

        windows = self.get_predict_windows(raster_source.get_extent())

        def predict_batch(chips, windows):
            nonlocal labels
            chips = np.array(chips)
            batch_labels = backend.predict(chips, windows)
            batch_labels = self.post_process_batch(windows, chips,
                                                   batch_labels)
            labels += batch_labels

            print('.' * len(chips), end='')

        batch_chips, batch_windows = [], []
        for window in windows:
            chip = raster_source.get_chip(window)
            if np.any(chip):
                batch_chips.append(chip)
                batch_windows.append(window)

            # Predict on batch
            if len(batch_chips) >= self.config.predict_batch_sz:
                predict_batch(batch_chips, batch_windows)
                batch_chips, batch_windows = [], []
        print()

        # Predict on remaining batch
        if len(batch_chips) > 0:
            predict_batch(batch_chips, batch_windows)

        return self.post_process_predictions(labels, scene)

    def eval(self):
        class_config = self.config.dataset.class_config
        scenes = [
            s.build(class_config, self.tmp_dir)
            for s in self.config.dataset.validation_scenes
        ]
        evaluators = [e.build(class_config) for e in self.config.evaluators]
        for evaluator in evaluators:
            log.info('Running evaluator: {}...'.format(
                type(evaluator).__name__))
            evaluator.process(scenes, self.tmp_dir)

    def bundle(self):
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as tmp_dir:
            for fn in self.config.backend.get_bundle_filenames():
                download_or_copy(
                    join(self.config.train_uri, fn), join(tmp_dir, fn))

            for a in self.config.analyzers:
                for fn in a.get_bundle_filenames():
                    download_or_copy(
                        join(self.config.analyze_uri, fn), join(tmp_dir, fn))

            download_or_copy(self.config.get_config_uri(),
                             join(tmp_dir, 'pipeline.json'))

            model_bundle_uri = join(self.config.bundle_uri, 'model-bundle.zip')
            model_bundle_path = get_local_path(model_bundle_uri, self.tmp_dir)
            zipdir(tmp_dir, model_bundle_path)
            upload_or_copy(model_bundle_path, model_bundle_uri)
