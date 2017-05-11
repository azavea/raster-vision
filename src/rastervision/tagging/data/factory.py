from os.path import join
import argparse

from rastervision.common.utils import _makedirs
from rastervision.common.settings import (
    datasets_path, results_path, TRAIN, VALIDATION)

from .planet_kaggle import PLANET_KAGGLE, TIFF, PlanetKaggleTiffFileGenerator


def get_data_generator(options):
    if options.dataset_name == PLANET_KAGGLE:
        if options.generator_name == TIFF:
            return PlanetKaggleTiffFileGenerator(
                datasets_path, options.active_input_inds, options.train_ratio,
                options.cross_validation)
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
            self.train_ratio = 0.8
            self.cross_validation = None

    options = Options()
    generator = get_data_generator(options)

    viz_path = join(
        results_path, 'gen_samples', dataset_name, generator_name, split)
    _makedirs(viz_path)

    gen = generator.make_split_generator(
        split, batch_size=batch_size, shuffle=True,
        augment=True, normalize=True, only_xy=False)

    for batch_ind in range(nb_batches):
        batch = next(gen)
        for sample_ind in range(batch_size):
            file_path = join(
                viz_path, '{}_{}.pdf'.format(batch_ind, sample_ind))
            generator.plot_sample(
                file_path,
                batch.all_x[sample_ind, :],
                batch.y[sample_ind, :],
                batch.file_inds[sample_ind])


def preprocess():
    PlanetKaggleTiffFileGenerator.preprocess(datasets_path)


def plot_generators():
    plot_generator(PLANET_KAGGLE, TIFF, TRAIN)
    plot_generator(PLANET_KAGGLE, TIFF, VALIDATION)


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
