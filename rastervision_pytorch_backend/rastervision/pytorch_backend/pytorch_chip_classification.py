from typing import Dict, Callable
from os.path import join
import uuid

import click

from rastervision.pipeline.file_system import (make_dir)
from rastervision.core.data import (ChipClassificationLabels, SceneConfig,
                                    DatasetConfig)
from rastervision.core.rv_pipeline import ChipClassificationPredictOptions
from rastervision.core.utils.misc import save_img
from rastervision.core.data_sample import DataSample
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend, chunk)
from rastervision.pytorch_learner import (
    ClassificationGeoDataConfig, GeoDataWindowConfig, get_base_datasets)


class PyTorchChipClassificationSampleWriter(PyTorchLearnerSampleWriter):
    def write_sample(self, sample: DataSample):
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
        class_name = self.class_config.names[class_id]
        class_dir = join(self.sample_dir, split_name, class_name)
        make_dir(class_dir)
        chip_path = join(class_dir, '{}-{}.png'.format(sample.scene_id,
                                                       self.sample_ind))
        save_img(sample.chip, chip_path)
        self.sample_ind += 1


class PyTorchChipClassification(PyTorchLearnerBackend):
    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, '{}.zip'.format(
            str(uuid.uuid4())))
        return PyTorchChipClassificationSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

    def _get_predictions(
            self,
            predict_options: ChipClassificationPredictOptions,
            hooks: Dict[str, Callable] = {}) -> ChipClassificationLabels:
        ds = get_base_datasets(self.learner.test_ds)[0]
        dl = self.learner.test_dl

        labels = ds.scene.label_store.empty_labels()

        windows = chunk(ds.windows, n=predict_options.batch_sz)
        preds = self.learner.iter_predictions(dl, raw_out=True)

        it = zip(windows, preds)
        with click.progressbar(it, length=len(dl), label='Predicting') as bar:
            for ws, (_, _, outs) in bar:
                pred_ids = self.learner.prob_to_pred(outs).numpy()
                outs = self.learner.output_to_numpy(outs)
                for w, class_id, class_scores in zip(ws, pred_ids, outs):
                    labels[w] = (class_id, class_scores)

        return labels

    def _make_pred_data_config(
            self,
            predict_options: ChipClassificationPredictOptions,
            scene_dataset: DatasetConfig,
            window_opts: GeoDataWindowConfig,
            scene: SceneConfig,
            hooks: Dict[str, Callable] = {}) -> ClassificationGeoDataConfig:
        cfg = self.learner.cfg.data
        return ClassificationGeoDataConfig(
            class_names=cfg.class_names,
            class_colors=cfg.class_colors,
            img_sz=cfg.img_sz,
            num_workers=cfg.num_workers,
            base_transform=cfg.base_transform,
            plot_options=cfg.plot_options,
            scene_dataset=scene_dataset,
            window_opts=window_opts)
