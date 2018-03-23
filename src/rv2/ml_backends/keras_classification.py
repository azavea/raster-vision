from os.path import join
import tempfile
import shutil

from rv2.core.ml_backend import MLBackend
from rv2.utils.files import (
    make_dir, get_local_path, upload_if_needed, download_if_needed,
    RV_TEMP_DIR)
from rv2.utils.misc import save_img


class TrainingPackage(object):
    """Represents paths for training data and associated utilities."""
    def __init__(self, base_uri):
        self.temp_dir_obj = tempfile.TemporaryDirectory(dir=RV_TEMP_DIR)
        self.temp_dir = self.temp_dir_obj.name

        self.base_uri = base_uri
        self.base_dir = self.get_local_path(base_uri)
        make_dir(self.base_dir, check_empty=True)

        self.training_uri = join(base_uri, 'training')
        make_dir(self.get_local_path(self.training_uri))
        self.training_zip_uri = join(base_uri, 'training.zip')

        self.validation_uri = join(base_uri, 'validation')
        make_dir(self.get_local_path(self.validation_uri))
        self.validation_zip_uri = join(base_uri, 'validation.zip')

    def get_local_path(self, uri):
        return get_local_path(uri, self.temp_dir)

    def upload_if_needed(self, uri):
        upload_if_needed(self.get_local_path(uri), uri)

    def download_if_needed(self, uri):
        return download_if_needed(uri, self.temp_dir)

    def download(self):
        pass

    def upload(self):
        self.upload_if_needed(self.training_zip_uri)
        self.upload_if_needed(self.validation_zip_uri)


class KerasClassification(MLBackend):
    def convert_training_data(self, training_data, validation_data, class_map,
                              options):
        """Convert training data to ImageFolder format.

        For each dataset, there is a directory for each class_name with chips
        of that class.
        """
        training_package = TrainingPackage(options.output_uri)
        training_dir = training_package.get_local_path(
            training_package.training_uri)
        validation_dir = training_package.get_local_path(
            training_package.validation_uri)

        def convert_dataset(dataset, output_dir):
            for class_name in class_map.get_class_names():
                class_dir = join(output_dir, class_name)
                make_dir(class_dir)

            for chip_ind, (chip, labels) in enumerate(dataset):
                class_id = labels.get_class_id()
                # If a chip is not associated with a class, don't
                # use it in training data.
                if class_id is not None:
                    class_name = class_map.get_by_id(class_id).name
                    chip_path = join(
                        output_dir, class_name, str(chip_ind) + '.png')
                    save_img(chip, chip_path)
            shutil.make_archive(output_dir, 'zip', output_dir)

        convert_dataset(training_data, training_dir)
        convert_dataset(validation_data, validation_dir)
        training_package.upload()

    def train(self, options):
        pass

    def predict(self, chip, options):
        pass
