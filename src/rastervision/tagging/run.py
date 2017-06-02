from rastervision.common.tasks.plot_curves import plot_curves, PLOT_CURVES
from rastervision.common.tasks.train_model import TRAIN_MODEL
from rastervision.common.tasks.validation_eval import VALIDATION_EVAL
from rastervision.common.tasks.aggregate_scores import aggregate_scores
from rastervision.common.run import Runner

from rastervision.tagging.data.factory import TaggingDataGeneratorFactory
from rastervision.tagging.models.factory import TaggingModelFactory
from rastervision.tagging.tasks.train_model import TaggingTrainModel
from rastervision.tagging.tasks.predict import (
    VALIDATION_PREDICT, TEST_PREDICT, validation_predict, test_predict)
from rastervision.tagging.tasks.validation_eval import validation_eval


class TaggingRunner(Runner):
    def __init__(self):
        self.valid_tasks = [
            TRAIN_MODEL, PLOT_CURVES, VALIDATION_PREDICT, VALIDATION_EVAL,
            TEST_PREDICT]
        self.model_factory_class = TaggingModelFactory
        self.data_generator_factory_class = TaggingDataGeneratorFactory

    def run_task(self, task):
        if task == TRAIN_MODEL:
            train_model = TaggingTrainModel(
                self.run_path, self.sync_results, self.options,
                self.generator, self.model)
            train_model.train_model()
        elif task == PLOT_CURVES:
            plot_curves(self.options)
        elif task == VALIDATION_PREDICT:
            validation_predict(
                self.run_path, self.model, self.options, self.generator)
        elif task == VALIDATION_EVAL:
            if self.options.aggregate_run_names is None:
                validation_eval(
                    self.run_path, self.model, self.options, self.generator)
            else:
                best_score_key = 'f2'
                aggregate_scores(self.options, best_score_key)
        elif task == TEST_PREDICT:
            test_predict(
                self.run_path, self.model, self.options, self.generator)
