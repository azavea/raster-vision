from os.path import join
import argparse

from rastervision.common.utils import _makedirs
from rastervision.common.settings import (
    datasets_path, results_path, TRAIN, VALIDATION)

from .potsdam import (
    POTSDAM, PotsdamImageFileGenerator, PotsdamNumpyFileGenerator)
from .vaihingen import (
    VAIHINGEN, VaihingenImageFileGenerator, VaihingenNumpyFileGenerator)
from .settings import NUMPY, IMAGE


def get_data_generator(options, datasets_path):
    if options.dataset_name == POTSDAM:
        if options.generator_name == NUMPY:
            return PotsdamNumpyFileGenerator(
                datasets_path, options.active_input_inds,
                options.train_ratio, options.cross_validation)
        elif options.generator_name == IMAGE:
            return PotsdamImageFileGenerator(
                datasets_path, options.active_input_inds,
                options.train_ratio, options.cross_validation)
        else:
            raise ValueError('{} is not a valid generator'.format(
                options.generator_name))
    elif options.dataset_name == VAIHINGEN:
        if options.generator_name == IMAGE:
            return VaihingenImageFileGenerator(
                datasets_path, options.active_input_inds,
                options.train_ratio, options.cross_validation)
        elif options.generator_name == NUMPY:
            return VaihingenNumpyFileGenerator(
                datasets_path, options.active_input_inds,
                options.train_ratio, options.cross_validation)
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
            self.active_input_inds = [0, 1, 2, 3]
            if dataset_name == POTSDAM:
                self.active_input_inds = [0, 1, 2, 3, 4]
            self.train_ratio = 0.8
            self.cross_validation = None

    options = Options()
    generator = get_data_generator(options, datasets_path)

    viz_path = join(
        results_path, 'gen_samples', dataset_name, generator_name, split)
    _makedirs(viz_path)

    gen = generator.make_split_generator(
        TRAIN, target_size=(400, 400), batch_size=batch_size, shuffle=True,
        augment=True, normalize=True, only_xy=False)

    for batch_ind in range(nb_batches):
        batch = next(gen)
        for sample_ind in range(batch_size):
            file_path = join(
                viz_path, '{}_{}.pdf'.format(batch_ind, sample_ind))
            generator.plot_sample(
                file_path,
                batch.all_x[sample_ind, :, :, :],
                batch.y[sample_ind, :, :, :])


def preprocess():
    VaihingenImageFileGenerator.preprocess(datasets_path)
    VaihingenNumpyFileGenerator.preprocess(datasets_path)

    PotsdamImageFileGenerator.preprocess(datasets_path)
    PotsdamNumpyFileGenerator.preprocess(datasets_path)


def plot_generators():
    plot_generator(VAIHINGEN, IMAGE, TRAIN)
    plot_generator(VAIHINGEN, IMAGE, VALIDATION)

    plot_generator(VAIHINGEN, NUMPY, TRAIN)
    plot_generator(VAIHINGEN, NUMPY, VALIDATION)

    plot_generator(POTSDAM, IMAGE, TRAIN)
    plot_generator(POTSDAM, IMAGE, VALIDATION)

    plot_generator(POTSDAM, NUMPY, TRAIN)
    plot_generator(POTSDAM, NUMPY, VALIDATION)


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
