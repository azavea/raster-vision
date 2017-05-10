from os.path import join
import sys

from rastervision.common.utils import Logger, make_sync_results, setup_run
from rastervision.common.tasks.plot_curves import plot_curves, PLOT_CURVES

from .options import SemsegOptions
from .data.settings import results_path, datasets_path, s3_bucket_name
from .data.factory import get_data_generator
from .models.factory import get_model, load_model
from .tasks.train_model import train_model, TRAIN_MODEL
from .tasks.validation_eval import validation_eval, VALIDATION_EVAL
from .tasks.predict import (
    validation_predict, test_predict, VALIDATION_PREDICT, TEST_PREDICT)
from .tasks.make_videos import MAKE_VIDEOS, make_videos

valid_tasks = [TRAIN_MODEL, PLOT_CURVES,
               VALIDATION_PREDICT, VALIDATION_EVAL, TEST_PREDICT,
               MAKE_VIDEOS]


def run_tasks(options_dict, tasks):
    """Run tasks specified on command line.

    This creates the RunOptions object from the json file specified on the
    command line, creates a data generator, and then runs the tasks.
    """
    options = SemsegOptions(options_dict)
    generator = get_data_generator(options, datasets_path)
    run_path = join(results_path, options.run_name)

    sync_results = make_sync_results(
        s3_bucket_name, options.run_name, run_path)
    setup_run(run_path, options, sync_results)
    sys.stdout = Logger(run_path)

    if len(tasks) == 0:
        tasks = valid_tasks

    for task in tasks:
        if task not in valid_tasks:
            raise ValueError('{} is not a valid task'.format(task))

    for task in tasks:
        if task == TRAIN_MODEL:
            model = get_model(
                run_path, options, generator, use_best=False)
            train_model(run_path, model, sync_results, options, generator)

            if options.train_stages:
                for stage in options.train_stages[1:]:
                    for key, value in stage.items():
                        if key == 'epochs':
                            options.epochs += value
                        else:
                            setattr(options, key, value)

                    model = get_model(
                        run_path, options, generator, use_best=False)
                    train_model(
                        run_path, model, sync_results, options, generator)
        elif task == PLOT_CURVES:
            plot_curves(run_path)
        elif task == VALIDATION_EVAL:
            model = load_model(
                run_path, options, generator, use_best=True)
            validation_eval(run_path, model, options, generator)
        elif task == TEST_PREDICT:
            model = load_model(
                run_path, options, generator, use_best=True)
            test_predict(run_path, model, options, generator)
        elif task == VALIDATION_PREDICT:
            model = load_model(
                run_path, options, generator, use_best=True)
            validation_predict(run_path, model, options, generator)
        elif task == MAKE_VIDEOS:
            make_videos(run_path, options, generator)

        sync_results()
