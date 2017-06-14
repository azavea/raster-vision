from os.path import isfile, join

from rastervision.common.settings import datasets_path, results_path
from rastervision.common.utils import s3_download


class ModelFactory():
    def __init__(self):
        self.datasets_path = datasets_path
        self.results_path = results_path

    def s3_download(self, run_name, file_name):
        s3_download(run_name, file_name)

    def make_model(self, options, generator):
        """Make a new model."""
        pass

    def load_model(self, run_path, options, generator, use_best=True):
        """Load an existing model."""
        # Load the model by weights. This permits loading weights from a saved
        # model into a model with a different architecture assuming the named
        # layers have compatible dimensions.
        model = self.make_model(options, generator)
        file_name = 'best_model.h5' if use_best else 'model.h5'
        model_path = join(run_path, file_name)
        # TODO raise exception if model_path doesn't exist
        model.load_weights(model_path, by_name=True)
        return model

    def get_model(self, run_path, options, generator, use_best=True):
        """Get a model by loading if it exists or making a new one."""
        model_path = join(run_path, 'model.h5')

        # Load the model if it's saved, or create a new one.
        if isfile(model_path):
            model = self.load_model(run_path, options, generator, use_best)
            print('Continuing training from saved model.')
        else:
            model = self.make_model(options, generator)
            print('Creating new model.')

        return model
