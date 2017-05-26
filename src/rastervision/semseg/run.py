from os.path import join
import sys

from rastervision.common.utils import Logger, make_sync_results, setup_run
from rastervision.common.tasks.plot_curves import plot_curves, PLOT_CURVES
from rastervision.common.tasks.train_model import TRAIN_MODEL
from rastervision.common.settings import results_path, datasets_path

from rastervision.semseg.data.factory import SemsegDataGeneratorFactory
from rastervision.semseg.models.factory import SemsegModelFactory
from rastervision.semseg.tasks.train_model import SemsegTrainModel
from rastervision.semseg.tasks.validation_eval import (
    validation_eval, VALIDATION_EVAL)
from rastervision.semseg.tasks.predict import (
    validation_predict, test_predict, VALIDATION_PREDICT, TEST_PREDICT)
from rastervision.semseg.tasks.make_videos import MAKE_VIDEOS, make_videos

valid_tasks = [TRAIN_MODEL, PLOT_CURVES,
               VALIDATION_PREDICT, VALIDATION_EVAL, TEST_PREDICT,
               MAKE_VIDEOS]


def run_tasks(options, tasks):
    """Run tasks specified on command line.

    This creates the RunOptions object from the json file specified on the
    command line, creates a data generator, and then runs the tasks.
    """
    model_factory = SemsegModelFactory()
    generator = SemsegDataGeneratorFactory().get_data_generator(options)
    run_path = join(results_path, options.run_name)

    sync_results = make_sync_results(options.run_name)
    setup_run(run_path, options, sync_results)
    sys.stdout = Logger(run_path)

    if len(tasks) == 0:
        tasks = valid_tasks

    for task in tasks:
        if task not in valid_tasks:
            raise ValueError('{} is not a valid task'.format(task))

    for task in tasks:
        if task == TRAIN_MODEL:
            model = model_factory.get_model(
                run_path, options, generator, use_best=False)
            train_model = SemsegTrainModel(
                run_path, sync_results, options, generator, model)
            train_model.train_model()

            if options.train_stages:
                for stage in options.train_stages[1:]:
                    for key, value in stage.items():
                        if key == 'epochs':
                            options.epochs += value
                        else:
                            setattr(options, key, value)

                    model = model_factory.get_model(
                        run_path, options, generator, use_best=False)
                    train_model = SemsegTrainModel(
                        run_path, sync_results, options, generator, model)
                    train_model.train_model()
        elif task == PLOT_CURVES:
            plot_curves(run_path)
        elif task == VALIDATION_EVAL:
            model = model_factory.load_model(
                run_path, options, generator, use_best=True)
            validation_eval(run_path, model, options, generator)
        elif task == TEST_PREDICT:
            model = model_factory.load_model(
                run_path, options, generator, use_best=True)
            test_predict(run_path, model, options, generator)
        elif task == VALIDATION_PREDICT:
            model = model_factory.load_model(
                run_path, options, generator, use_best=True)
            validation_predict(run_path, model, options, generator)
        elif task == MAKE_VIDEOS:
            make_videos(run_path, options, generator)

        sync_results()
