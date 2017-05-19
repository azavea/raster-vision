from os.path import join
import sys

from rastervision.common.utils import Logger, make_sync_results, setup_run
from rastervision.common.tasks.plot_curves import plot_curves, PLOT_CURVES
from rastervision.common.tasks.train_model import TRAIN_MODEL
from rastervision.common.settings import results_path

from rastervision.tagging.options import TaggingOptions
from rastervision.tagging.data.factory import get_data_generator
from rastervision.tagging.models.factory import TaggingModelFactory
from rastervision.tagging.tasks.train_model import TaggingTrainModel
from rastervision.tagging.tasks.predict import (
    VALIDATION_PREDICT, TEST_PREDICT, validation_predict, test_predict)
from rastervision.tagging.tasks.validation_eval import (
    VALIDATION_EVAL, validation_eval)

valid_tasks = [TRAIN_MODEL, PLOT_CURVES, VALIDATION_PREDICT,
               VALIDATION_EVAL, TEST_PREDICT]


def run_tasks(options_dict, tasks):
    """Run tasks specified on command line.

    This creates the RunOptions object from the json file specified on the
    command line, creates a data generator, and then runs the tasks.
    """
    options = TaggingOptions(options_dict)
    model_factory = TaggingModelFactory()
    generator = get_data_generator(options)
    run_path = join(results_path, options.run_name)

    sync_results = make_sync_results(options.run_name)
    setup_run(run_path, options, sync_results)
    sys.stdout = Logger(run_path)

    if len(tasks) == 0:
        tasks = valid_tasks

    for task in tasks:
        if task not in valid_tasks:
            raise ValueError('{} is not a valid task'.format(task))

    model = model_factory.get_model(
        run_path, options, generator, use_best=True)

    for task in tasks:
        if task == TRAIN_MODEL:
            train_model = TaggingTrainModel(
                run_path, sync_results, options, generator, model)
            train_model.train_model()
        elif task == PLOT_CURVES:
            plot_curves(run_path)
        elif task == VALIDATION_PREDICT:
            validation_predict(run_path, model, options, generator)
        elif task == VALIDATION_EVAL:
            validation_eval(run_path, generator)
        elif task == TEST_PREDICT:
            test_predict(run_path, model, options, generator)
        else:
            raise ValueError('{} is not a valid task'.format(task))

        sync_results()
