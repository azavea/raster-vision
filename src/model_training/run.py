"""
Execute a sequence of tasks for a run, given a json file with options for that
run. Example usage: `python run.py options.json setup train eval`
"""
import json
from os.path import join, isfile, isdir
import sys
import argparse
from subprocess import call

from .data.preprocess import _makedirs
from .data.settings import results_path, get_dataset_info, POTSDAM
from .train import make_model, train_model, CONV_LOGISTIC, FCN_RESNET
from .eval_run import eval_run

SETUP = 'setup'
TRAIN = 'train'
EVAL = 'eval'


class RunOptions():
    """ Represents the options used to control an experimental run. """
    def __init__(self, options):
        # Required options
        self.model_type = options['model_type']
        self.run_name = options['run_name']
        self.dataset = options['dataset']
        self.include_ir = options['include_ir']
        self.include_depth = options['include_depth']
        self.include_ndvi = options['include_ndvi']

        self.batch_size = options['batch_size']
        self.nb_epoch = options['nb_epoch']
        self.samples_per_epoch = options['samples_per_epoch']
        self.nb_val_samples = options['nb_val_samples']

        self.git_commit = options['git_commit']

        # Optional options
        self.patience = options.get('patience')
        self.lr_schedule = options.get('lr_schedule')
        # Controls how many samples to use in the final evaluation of the run.
        self.nb_eval_samples = options.get('nb_eval_samples')

        # model_type dependent options
        if self.model_type == CONV_LOGISTIC:
            self.kernel_size = options['kernel_size']
        elif self.model_type == FCN_RESNET:
            self.drop_prob = options['drop_prob']
            self.is_big_model = options['is_big_model']

        # dataset dependent options
        if self.dataset == POTSDAM and 'sharah_train_ratio' in options:
            dataset_info = get_dataset_info(POTSDAM)
            self.train_ratio = dataset_info.sharah_train_ratio
        else:
            self.train_ratio = options['train_ratio']


def load_options(file_path):
    options = None
    with open(file_path) as options_file:
        options = json.load(options_file)
        options = RunOptions(options)

    return options


class Logger(object):
    """ Used to log stdout to a file and to the console. """
    def __init__(self, run_path):
        self.terminal = sys.stdout
        self.log = open(join(run_path, 'stdout.txt'), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def setup_run(options, sync_results):
    """ Setup directory for the results of the run """
    # If there isn't a run directory, try to download it from S3.
    run_path = join(results_path, options.run_name)
    if not isdir(run_path):
        sync_results(download=True)

    # Ensure there is a run directory.
    _makedirs(run_path)

    # Read the options file and write it to the run directory.
    options_json = json.dumps(options.__dict__, sort_keys=True, indent=4)
    options_path = join(run_path, 'options.json')
    with open(options_path, 'w') as options_file:
        options_file.write(options_json)

    sys.stdout = Logger(run_path)


def load_model(options, dataset_info, run_path, use_best=False):
    # Load the model by weights. This permits loading weights from a saved
    # model into a model with a different architecture assuming the named
    # layers have compatible dimensions.
    model = make_model(options, dataset_info)
    file_name = 'best_model.h5' if use_best else 'model.h5'
    model.load_weights(join(run_path, file_name), by_name=True)
    return model


def train_run(options, dataset_info, run_path, sync_results):
    model_path = join(run_path, 'model.h5')

    # Load the model if it's saved, or create a new one.
    if isfile(model_path):
        model = load_model(options, dataset_info, run_path)
        print('Continuing training on {}'.format(model_path))
    else:
        model = make_model(options, dataset_info)
        print('Creating new model.')
    train_model(model, sync_results, options, dataset_info)

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', nargs='?',
                        help='path to the options json file')
    parser.add_argument('tasks', nargs='*', help='list of tasks to perform',
                        default=['setup', 'train', 'eval'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    options = load_options(args.file_path)
    dataset_info = get_dataset_info(options.dataset)
    dataset_info.setup(
        include_ir=options.include_ir, include_depth=options.include_depth,
        include_ndvi=options.include_ndvi, train_ratio=options.train_ratio)
    run_path = join(results_path, options.run_name)

    def sync_results(download=False):
        s3_run_path = 's3://otid-data/results/{}'.format(options.run_name)
        if download:
            call(['aws', 's3', 'sync', s3_run_path, run_path])
        else:
            call(['aws', 's3', 'sync', run_path, s3_run_path])

    valid_tasks = [SETUP, TRAIN, EVAL]
    for task in args.tasks:
        if task not in valid_tasks:
            raise ValueError('{} is not a valid task'.format(task))

    for task in args.tasks:
        if task == SETUP:
            setup_run(options, sync_results)
        elif task == TRAIN:
            train_run(options, dataset_info, run_path, sync_results)
        elif task == EVAL:
            model = load_model(options, dataset_info, run_path, use_best=True)
            eval_run(model, options, dataset_info)
            sync_results()
