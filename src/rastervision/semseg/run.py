from rastervision.common.tasks.plot_curves import plot_curves, PLOT_CURVES
from rastervision.common.tasks.train_model import TRAIN_MODEL
from rastervision.common.tasks.aggregate_scores import aggregate_scores
from rastervision.common.tasks.validation_eval import VALIDATION_EVAL
from rastervision.common.run import Runner

from rastervision.semseg.data.factory import SemsegDataGeneratorFactory
from rastervision.semseg.models.factory import SemsegModelFactory
from rastervision.semseg.tasks.train_model import SemsegTrainModel
from rastervision.semseg.tasks.validation_eval import validation_eval
from rastervision.semseg.tasks.predict import (
    validation_predict, test_predict, VALIDATION_PREDICT, TEST_PREDICT)
from rastervision.semseg.tasks.make_videos import MAKE_VIDEOS, make_videos


class SemsegRunner(Runner):
    def __init__(self):
        self.valid_tasks = [
            TRAIN_MODEL, PLOT_CURVES, VALIDATION_PREDICT, VALIDATION_EVAL,
            TEST_PREDICT, MAKE_VIDEOS]

        self.model_factory_class = SemsegModelFactory
        self.data_generator_factory_class = SemsegDataGeneratorFactory

    def run_task(self, task):
        if task == TRAIN_MODEL:
            train_model = SemsegTrainModel(
                self.run_path, self.sync_results, self.options, self.generator,
                self.model)
            train_model.train_model()

            if self.options.train_stages:
                for stage in self.options.train_stages[1:]:
                    for key, value in stage.items():
                        if key == 'epochs':
                            self.options.epochs += value
                        else:
                            setattr(self.options, key, value)

                    self.model = self.model_factory.get_model(
                        self.run_path, self.options, self.generator,
                        use_best=False)
                    train_model = SemsegTrainModel(
                        self.run_path, self.sync_results, self.options,
                        self.generator, self.model)
                    train_model.train_model()
        elif task == PLOT_CURVES:
            plot_curves(self.options)
        elif task == VALIDATION_EVAL:
            if self.options.aggregate_run_names is None:
                validation_eval(
                    self.run_path, self.model, self.options, self.generator)
            else:
                best_score_key = 'avg_accuracy'
                aggregate_scores(self.options, best_score_key)
        elif task == TEST_PREDICT:
            test_predict(
                self.run_path, self.model, self.options, self.generator)
        elif task == VALIDATION_PREDICT:
            validation_predict(
                self.run_path, self.model, self.options, self.generator)
        elif task == MAKE_VIDEOS:
            make_videos(self.run_path, self.options, self.generator)
