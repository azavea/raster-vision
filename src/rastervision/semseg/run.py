from rastervision.common.tasks.plot_curves import plot_curves, PLOT_CURVES
from rastervision.common.tasks.train_model import TRAIN_MODEL
from rastervision.common.tasks.aggregate_scores import aggregate_scores
from rastervision.common.run import Runner
from rastervision.common.options import AGG_SUMMARY

from rastervision.semseg.options import SemsegOptions
from rastervision.semseg.data.factory import SemsegDataGeneratorFactory
from rastervision.semseg.models.factory import SemsegModelFactory
from rastervision.semseg.tasks.train_model import SemsegTrainModel
from rastervision.semseg.tasks.validation_eval import (
    validation_eval, VALIDATION_EVAL)
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
        self.options_class = SemsegOptions
        self.agg_file_names = [
            'log.txt', 'options.json', 'scores.json']

    def run_task(self, task):
        aggregate_type = self.options.aggregate_type
        if task == TRAIN_MODEL:
            if aggregate_type is None:
                train_model = SemsegTrainModel(
                    self.run_path, self.sync_results, self.options,
                    self.generator, self.model)
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

                self.model = self.model_factory.get_model(
                    self.run_path, self.options, self.generator,
                    use_best=self.options.use_best_model)
        elif task == PLOT_CURVES:
            plot_curves(self.options)
        elif task == VALIDATION_EVAL:
            if aggregate_type == AGG_SUMMARY:
                best_score_key = 'avg_accuracy'
                aggregate_scores(self.options, best_score_key)
            else:
                validation_eval(
                    self.run_path, self.model, self.options, self.generator)
        elif task == TEST_PREDICT:
            if aggregate_type is None:
                test_predict(
                    self.run_path, self.model, self.options, self.generator)
        elif task == VALIDATION_PREDICT:
            if aggregate_type is None:
                validation_predict(
                    self.run_path, self.model, self.options, self.generator)
        elif task == MAKE_VIDEOS:
            if aggregate_type is None:
                make_videos(self.run_path, self.options, self.generator)
