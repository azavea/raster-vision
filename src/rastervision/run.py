"""
Execute a sequence of tasks for a run, given a json file with options for that
run. Example usage: `python run.py options.json train_model`
"""
import argparse
import json

from rastervision.semseg.settings import SEMSEG
from rastervision.semseg.run import run_tasks as semseg_run_tasks
from rastervision.tagging.settings import TAGGING
from rastervision.tagging.run import run_tasks as tagging_run_tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', nargs='?',
                        help='path to the options json file')
    parser.add_argument('tasks', nargs='*', help='list of tasks to perform')
    return parser.parse_args()


def run_tasks():
    """Run tasks specified on command line.

    This creates the RunOptions object from the json file specified on the
    command line, creates a data generator, and then runs the tasks.
    """
    args = parse_args()
    with open(args.file_path) as options_file:
        options_dict = json.load(options_file)
        problem_type = options_dict.get('problem_type')

        if problem_type is None:
            raise ValueError('problem_type must be specified in options file')

        if problem_type == SEMSEG:
            semseg_run_tasks(options_dict, args.tasks)
        elif problem_type == TAGGING:
            tagging_run_tasks(options_dict, args.tasks)
        else:
            raise ValueError('{} is not a valid problem_type'.format(
                problem_type))


if __name__ == '__main__':
    run_tasks()
