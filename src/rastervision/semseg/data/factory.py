from os.path import join

from rastervision.common.data.factory import DataGeneratorFactory
from rastervision.common.data.generators import ROTATE90, HFLIP, VFLIP
from rastervision.common.utils import _makedirs

from rastervision.semseg.data.potsdam import (
    POTSDAM, PotsdamImageFileGenerator, PotsdamNumpyFileGenerator)
from rastervision.semseg.data.vaihingen import (
    VAIHINGEN, VaihingenImageFileGenerator, VaihingenNumpyFileGenerator)
from rastervision.semseg.data.settings import NUMPY, IMAGE


class SemsegDataGeneratorFactory(DataGeneratorFactory):
    def __init__(self):
        super().__init__([POTSDAM, VAIHINGEN], [IMAGE, NUMPY])

    def get_class(self, dataset_name, generator_name):
        self.validate_keys(dataset_name, generator_name)
        if dataset_name == POTSDAM:
            if generator_name == NUMPY:
                return PotsdamNumpyFileGenerator
            elif generator_name == IMAGE:
                return PotsdamImageFileGenerator
        elif dataset_name == VAIHINGEN:
            if generator_name == IMAGE:
                return VaihingenImageFileGenerator
            elif generator_name == NUMPY:
                return VaihingenNumpyFileGenerator

    def plot_generator(self, dataset_name, generator_name, split):
        self.validate_keys(dataset_name, generator_name, split)
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
                self.augment_methods = [HFLIP, VFLIP, ROTATE90]

        options = Options()
        generator = self.get_data_generator(options)

        viz_path = join(
            self.results_path, 'gen_samples', dataset_name, generator_name,
            split)
        _makedirs(viz_path)

        gen = generator.make_split_generator(
            split, target_size=(400, 400), batch_size=batch_size, shuffle=True,
            augment_methods=options.augment_methods, normalize=True,
            only_xy=False)

        for batch_ind in range(nb_batches):
            batch = next(gen)
            for sample_ind in range(batch_size):
                file_path = join(
                    viz_path, '{}_{}.pdf'.format(batch_ind, sample_ind))
                generator.plot_sample(
                    file_path,
                    batch.all_x[sample_ind, :, :, :],
                    batch.y[sample_ind, :, :, :])


if __name__ == '__main__':
    factory = SemsegDataGeneratorFactory()
    factory.run()
