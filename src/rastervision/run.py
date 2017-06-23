"""
Execute a sequence of tasks for a run, given a json file with options for that
run. Example usage: `python run.py options.json train_model`
"""
import argparse
import json

from rastervision.semseg.settings import SEMSEG
from rastervision.semseg.run import SemsegRunner

from rastervision.tagging.settings import TAGGING
from rastervision.tagging.run import TaggingRunner


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
        problem_type = json.load(options_file)['problem_type']
        if problem_type == SEMSEG:
            runner = SemsegRunner()
        elif problem_type == TAGGING:
            runner = TaggingRunner()
        else:
            raise ValueError('{} is not a valid problem_type'.format(
                problem_type))
        runner.run_tasks(args.file_path, args.tasks)


if __name__ == '__main__':
    run_tasks()
