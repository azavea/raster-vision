from os.path import join
import os
import tempfile
import shutil
from urllib.parse import urlparse

import numpy as np
from keras_classification.commands.train import _train
from keras_classification.protos.pipeline_pb2 import PipelineConfig
from keras_classification.builders import model_builder
from keras_classification.utils import predict
from google.protobuf import json_format

from rastervision.core.ml_backend import MLBackend
from rastervision.utils.files import (
    make_dir, get_local_path, upload_if_needed, download_if_needed,
    RV_TEMP_DIR, start_sync, sync_dir, load_json_config)
from rastervision.utils.misc import save_img
from rastervision.labels.classification_labels import ClassificationLabels
from rastervision.core.box import Box


class FileGroup(object):
    def __init__(self, base_uri):
        self.temp_dir_obj = tempfile.TemporaryDirectory(dir=RV_TEMP_DIR)
        self.temp_dir = self.temp_dir_obj.name

        self.base_uri = base_uri
        self.base_dir = self.get_local_path(base_uri)
        make_dir(self.base_dir)

    def get_local_path(self, uri):
        return get_local_path(uri, self.temp_dir)

    def upload_if_needed(self, uri):
        upload_if_needed(self.get_local_path(uri), uri)

    def download_if_needed(self, uri):
        return download_if_needed(uri, self.temp_dir)


class DatasetFiles(FileGroup):
    """Utilities for files produced when calling convert_training_data."""
    def __init__(self, base_uri):
        FileGroup.__init__(self, base_uri)

        self.training_uri = join(base_uri, 'training')
        make_dir(self.get_local_path(self.training_uri))
        self.training_zip_uri = join(base_uri, 'training.zip')

        self.validation_uri = join(base_uri, 'validation')
        make_dir(self.get_local_path(self.validation_uri))
        self.validation_zip_uri = join(base_uri, 'validation.zip')

    def download(self):
        def _download(data_zip_uri):
            data_zip_path = self.download_if_needed(data_zip_uri)
            data_dir = os.path.splitext(data_zip_path)[0]
            shutil.unpack_archive(data_zip_path, data_dir)

        _download(self.training_zip_uri)
        _download(self.validation_zip_uri)

    def upload(self):
        def _upload(data_uri):
            data_dir = self.get_local_path(data_uri)
            shutil.make_archive(data_dir, 'zip', data_dir)
            self.upload_if_needed(data_uri + '.zip')

        _upload(self.training_uri)
        _upload(self.validation_uri)


class ModelFiles(FileGroup):
    """Utilities for files produced when calling train."""
    def __init__(self, base_uri):
        FileGroup.__init__(self, base_uri)

        self.model_uri = join(self.base_uri, 'model')
        self.log_uri = join(self.base_uri, 'log.csv')

    def download_backend_config(self, backend_config_uri,
                                dataset_files, class_map):
        config = load_json_config(backend_config_uri, PipelineConfig())

        # Update config using local paths.
        config.trainer.options.output_dir = self.get_local_path(self.base_uri)
        config.model.model_path = self.get_local_path(self.model_uri)

        config.trainer.options.training_data_dir = \
            dataset_files.get_local_path(dataset_files.training_uri)
        config.trainer.options.validation_data_dir = \
            dataset_files.get_local_path(dataset_files.validation_uri)

        del config.trainer.options.class_names[:]
        config.trainer.options.class_names.extend(
            class_map.get_class_names())

        # Save an updated copy of the config file.
        config_path = self.get_local_path(backend_config_uri)
        config_str = json_format.MessageToJson(config)
        with open(config_path, 'w') as config_file:
            config_file.write(config_str)
        return config_path


class KerasClassification(MLBackend):
    def __init__(self):
        self.model = None

    def convert_training_data(self, training_data, validation_data, class_map,
                              options):
        """Convert training data to ImageFolder format.

        For each dataset, there is a directory for each class_name with chips
        of that class.
        """
        dataset_files = DatasetFiles(options.output_uri)
        training_dir = dataset_files.get_local_path(
            dataset_files.training_uri)
        validation_dir = dataset_files.get_local_path(
            dataset_files.validation_uri)

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

        convert_dataset(training_data, training_dir)
        convert_dataset(validation_data, validation_dir)
        dataset_files.upload()

    def train(self, class_map, options):
        dataset_files = DatasetFiles(options.training_data_uri)
        dataset_files.download()

        model_files = ModelFiles(options.output_uri)
        backend_config_path = model_files.download_backend_config(
            options.backend_config_uri, dataset_files, class_map)

        start_sync(model_files.base_dir, options.output_uri,
                   sync_interval=options.sync_interval)
        _train(backend_config_path)

        if urlparse(options.output_uri).scheme == 's3':
            sync_dir(model_files.base_dir, options.output_uri, delete=True)

    def predict(self, chip, options):
        if self.model is None:
            with tempfile.TemporaryDirectory(dir=RV_TEMP_DIR) as temp_dir:
                model_path = download_if_needed(options.model_uri, temp_dir)
                self.model = model_builder.build_from_path(model_path)

        # Make a batch of size 1. This won't be needed once we refactor
        # the predict method to take batches of chips.
        batch = np.expand_dims(chip, axis=0)
        probs = predict(batch, self.model)
        # Add 1 to class_id since they start at 1.
        class_id = int(np.argmax(probs[0]) + 1)

        # Make labels with a single dummy cell. 
        labels = ClassificationLabels()
        dummy_cell = Box(0, 0, 0, 0)
        labels.set_cell(dummy_cell, class_id)

        return labels
