from os.path import join
import argparse

from .potsdam import (
    POTSDAM, PotsdamImageFileGenerator, PotsdamNumpyFileGenerator)
from .vaihingen import (
    VAIHINGEN, VaihingenImageFileGenerator, VaihingenNumpyFileGenerator)
from .generators import NUMPY, IMAGE, TRAIN, VALIDATION
from .utils import _makedirs, plot_sample
from .settings import datasets_path, results_path


def get_data_generator(options, datasets_path):
    if options.dataset_name == POTSDAM:
        if options.generator_name == NUMPY:
            return PotsdamNumpyFileGenerator(
                datasets_path, options.include_ir,
                options.include_depth, options.include_ndvi,
                options.train_ratio)
        elif options.generator_name == IMAGE:
            return PotsdamImageFileGenerator(
                datasets_path, options.include_ir,
                options.include_depth, options.include_ndvi,
                options.train_ratio)
        else:
            raise ValueError('{} is not a valid generator'.format(
                options.generator_name))
    elif options.dataset_name == VAIHINGEN:
        if options.generator_name == IMAGE:
            return VaihingenImageFileGenerator(
                datasets_path,
                options.include_depth, options.include_ndvi,
                options.train_ratio)
        elif options.generator_name == NUMPY:
            return VaihingenNumpyFileGenerator(
                datasets_path,
                options.include_depth, options.include_ndvi,
                options.train_ratio)
        else:
            raise ValueError('{} is not a valid generator'.format(
                options.generator_name))
    else:
        raise ValueError('{} is not a valid dataset'.format(
            options.dataset_name))


def plot_generator(dataset_name, generator_name, split):
    nb_batches = 2
    batch_size = 4

    class Options():
        def __init__(self):
            self.dataset_name = dataset_name
            self.generator_name = generator_name
            self.include_ir = True
            self.include_depth = True
            self.include_ndvi = True
            self.train_ratio = 0.7

    options = Options()
    generator = get_data_generator(options, datasets_path)

    viz_path = join(
        results_path, 'gen_samples', dataset_name, generator_name, split)
    _makedirs(viz_path)

    gen = generator.make_split_generator(
        TRAIN, tile_size=(400, 400), batch_size=batch_size, shuffle=True,
        augment=True, normalize=True, eval_mode=True)

    for batch_ind in range(nb_batches):
        inputs, outputs, _, _ = next(gen)
        for sample_ind in range(batch_size):
            file_path = join(
                viz_path, '{}_{}.pdf'.format(batch_ind, sample_ind))
            plot_sample(
                file_path, inputs[sample_ind, :, :, :],
                outputs[sample_ind, :, :, :], generator)


def preprocess():
    PotsdamImageFileGenerator.preprocess(datasets_path)
    PotsdamNumpyFileGenerator.preprocess(datasets_path)

    VaihingenImageFileGenerator.preprocess(datasets_path)
    VaihingenNumpyFileGenerator.preprocess(datasets_path)


def plot_generators():
    plot_generator(POTSDAM, IMAGE, TRAIN)
    plot_generator(POTSDAM, IMAGE, VALIDATION)

    plot_generator(POTSDAM, NUMPY, TRAIN)
    plot_generator(POTSDAM, NUMPY, VALIDATION)

    plot_generator(VAIHINGEN, IMAGE, TRAIN)
    plot_generator(VAIHINGEN, IMAGE, VALIDATION)

    plot_generator(VAIHINGEN, NUMPY, TRAIN)
    plot_generator(VAIHINGEN, NUMPY, VALIDATION)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--preprocess',
        action='store_true', help='run preprocessing for all generators')
    parser.add_argument(
        '--plot',
        action='store_true', help='plot all generators')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.preprocess:
        preprocess()

    if args.plot:
        plot_generators()
