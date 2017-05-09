"""
Execute a sequence of tasks for a run, given a json file with options for that
run. Example usage: `python run.py options.json setup_run train_model`
"""
import sys
import argparse
from subprocess import call
from os.path import join

from .options import load_options

from .data.settings import results_path, datasets_path, s3_bucket_name
from .data.factory import get_data_generator
from .data.utils import _makedirs

from .models.factory import get_model, load_model

from .tasks.setup_run import setup_run
from .tasks.train_model import train_model, TRAIN_MODEL
from .tasks.plot_curves import plot_curves, PLOT_CURVES
from .tasks.validation_eval import validation_eval, VALIDATION_EVAL
from .tasks.predict import (
    validation_predict, test_predict, VALIDATION_PREDICT, TEST_PREDICT)
from .tasks.make_videos import MAKE_VIDEOS, make_videos

valid_tasks = [TRAIN_MODEL, PLOT_CURVES,
               VALIDATION_PREDICT, VALIDATION_EVAL, TEST_PREDICT,
               MAKE_VIDEOS]


class Logger(object):
    """Used to log stdout to a file and to the console."""

    def __init__(self, run_path):
        self.terminal = sys.stdout
        self.log = open(join(run_path, 'stdout.txt'), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', nargs='?',
                        help='path to the options json file')
    parser.add_argument('tasks', nargs='*', help='list of tasks to perform',
                        default=valid_tasks)
    return parser.parse_args()


def run_tasks():
    """Run tasks specified on command line.

    This creates the RunOptions object from the json file specified on the
    command line, creates a data generator, and then runs the tasks.
    """
    args = parse_args()
    options = load_options(args.file_path)
    generator = get_data_generator(options, datasets_path)
    run_path = join(results_path, options.run_name)

    def sync_results(download=False):
        s3_run_path = 's3://{}/results/{}'.format(
            s3_bucket_name, options.run_name)
        if download:
            call(['aws', 's3', 'sync', s3_run_path, run_path])
        else:
            call(['aws', 's3', 'sync', run_path, s3_run_path])

    setup_run(run_path, options, sync_results)
    sys.stdout = Logger(run_path)

    # Run the tasks specified on the command line.
    for task in args.tasks:
        if task not in valid_tasks:
            raise ValueError('{} is not a valid task'.format(task))

    for task in args.tasks:
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


if __name__ == '__main__':
    run_tasks()
