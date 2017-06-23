import argparse

from rastervision.common.settings import (
    split_names, datasets_path, results_path, TRAIN, VALIDATION)

PLOT = 'plot'
PREPROCESS = 'preprocess'
ALL = 'all'


class DataGeneratorFactory():
    def __init__(self, dataset_names, generator_names):
        self.dataset_names = dataset_names
        self.generator_names = generator_names

        self.datasets_path = datasets_path
        self.results_path = results_path

    def validate_keys(self, dataset_name, generator_name, split=None):
        if dataset_name not in self.dataset_names:
            raise ValueError(
                '{} is not a valid dataset_name'.format(dataset_name))

        if generator_name not in self.generator_names:
            raise ValueError(
                '{} is not a valid generator_name'.format(generator_name))

        if split is not None:
            if split not in split_names:
                raise ValueError(
                    '{} is not a valid split'.format(split))

    def get_data_generator(self, options):
        data_generator_class = \
            self.get_class(options.dataset_name, options.generator_name)
        return data_generator_class(
            self.datasets_path, options)

    def preprocess(self, dataset_name, generator_name):
        data_generator_class = \
            self.get_class(dataset_name, generator_name)
        data_generator_class.preprocess(self.datasets_path)

    def get_class(self, dataset_name, generator_name):
        raise NotImplementedError()

    def plot_generator(self, dataset_name, generator_name, split):
        raise NotImplementedError()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            'dataset_name', choices=self.dataset_names + [ALL])
        parser.add_argument(
            'generator_name', choices=self.generator_names + [ALL])
        parser.add_argument(
            'task', help='task to run on data generator',
            choices=[PREPROCESS, PLOT, ALL])

        return parser.parse_args()

    def run(self):
        args = self.parse_args()

        dataset_names = self.dataset_names if args.dataset_name == ALL \
            else [args.dataset_name]
        generator_names = self.generator_names if args.generator_name == ALL \
            else [args.generator_name]
        tasks = [PREPROCESS, PLOT] if args.task == ALL else [args.task]

        for dataset_name in dataset_names:
            for generator_name in generator_names:
                for task in tasks:
                    if task == PREPROCESS:
                        self.preprocess(dataset_name, generator_name)
                    elif task == PLOT:
                        self.plot_generator(
                            dataset_name, generator_name, TRAIN)
                        self.plot_generator(
                            dataset_name, generator_name, VALIDATION)
