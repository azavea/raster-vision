from rastervision.common.settings import TRAIN, VALIDATION, TEST
from rastervision.common.tasks.plot_curves import plot_curves, PLOT_CURVES
from rastervision.common.tasks.train_model import TRAIN_MODEL
from rastervision.common.tasks.aggregate_scores import aggregate_scores
from rastervision.common.run import Runner
from rastervision.common.options import (
    AGG_SUMMARY, AGG_ENSEMBLE, AGG_CONCAT)

from rastervision.tagging.data.factory import TaggingDataGeneratorFactory
from rastervision.tagging.options import TaggingOptions
from rastervision.tagging.models.factory import TaggingModelFactory
from rastervision.tagging.tasks.train_model import TaggingTrainModel
from rastervision.tagging.tasks.predict import (
    TRAIN_PROBS, TRAIN_PREDICT, VALIDATION_PROBS, VALIDATION_PREDICT,
    TEST_PROBS, TEST_PREDICT, compute_ensemble_probs, compute_probs,
    compute_preds, compute_concat_probs)
from rastervision.tagging.tasks.validation_eval import (
    VALIDATION_EVAL, validation_eval)
from rastervision.tagging.tasks.train_thresholds import (
    TRAIN_THRESHOLDS, train_thresholds)


class TaggingRunner(Runner):
    def __init__(self):
        self.valid_tasks = [
            TRAIN_MODEL, PLOT_CURVES,
            TRAIN_PROBS, VALIDATION_PROBS, TEST_PROBS,
            TRAIN_THRESHOLDS,
            TRAIN_PREDICT, VALIDATION_PREDICT, TEST_PREDICT,
            VALIDATION_EVAL]
        self.model_factory_class = TaggingModelFactory
        self.data_generator_factory_class = TaggingDataGeneratorFactory
        self.options_class = TaggingOptions
        self.agg_file_names = [
            'log.txt', 'options.json', 'scores.json', 'train_probs.npy',
            'test_probs.npy', 'validation_probs.npy']

    def run_task(self, task):
        aggregate_type = self.options.aggregate_type
        if task == TRAIN_MODEL:
            if aggregate_type is None:
                train_model = TaggingTrainModel(
                    self.run_path, self.sync_results, self.options,
                    self.generator, self.model)
                train_model.train_model()
        elif task == PLOT_CURVES:
            if aggregate_type in [None, AGG_SUMMARY]:
                plot_curves(self.options)
        elif task == TRAIN_PROBS:
            if aggregate_type == AGG_ENSEMBLE:
                compute_ensemble_probs(
                    self.run_path, self.options, self.generator, TRAIN)
            elif aggregate_type == AGG_CONCAT:
                compute_concat_probs(
                    self.run_path, self.options, self.generator, TRAIN)
            elif aggregate_type is None:
                compute_probs(self.run_path, self.model, self.options,
                              self.generator, TRAIN)
        elif task == VALIDATION_PROBS:
            if aggregate_type == AGG_ENSEMBLE:
                compute_ensemble_probs(
                    self.run_path, self.options, self.generator, VALIDATION)
            elif aggregate_type == AGG_CONCAT:
                compute_concat_probs(
                    self.run_path, self.options, self.generator, VALIDATION)
            elif aggregate_type is None:
                compute_probs(self.run_path, self.model, self.options,
                              self.generator, VALIDATION)
        elif task == TEST_PROBS:
            if aggregate_type == AGG_ENSEMBLE:
                compute_ensemble_probs(
                    self.run_path, self.options, self.generator, TEST)
            elif aggregate_type == AGG_CONCAT:
                compute_concat_probs(
                    self.run_path, self.options, self.generator, TEST)
            elif aggregate_type is None:
                compute_probs(self.run_path, self.model, self.options,
                              self.generator, TEST)
        elif task == TRAIN_THRESHOLDS:
            if aggregate_type in [None, AGG_ENSEMBLE, AGG_CONCAT]:
                train_thresholds(
                    self.run_path, self.options, self.generator)
        elif task == TRAIN_PREDICT:
            if aggregate_type in [None, AGG_ENSEMBLE, AGG_CONCAT]:
                compute_preds(
                    self.run_path, self.options, self.generator, TRAIN)
        elif task == VALIDATION_PREDICT:
            if aggregate_type in [None, AGG_ENSEMBLE, AGG_CONCAT]:
                compute_preds(self.run_path, self.options,
                              self.generator, VALIDATION)
        elif task == TEST_PREDICT:
            if aggregate_type in [None, AGG_ENSEMBLE, AGG_CONCAT]:
                compute_preds(
                    self.run_path, self.options, self.generator, TEST)
        elif task == VALIDATION_EVAL:
            if aggregate_type == AGG_SUMMARY:
                best_score_key = 'f2'
                aggregate_scores(self.options, best_score_key)
            else:
                validation_eval(
                    self.run_path, self.options, self.generator)
