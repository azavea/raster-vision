import os
import shutil

from google.protobuf import json_format


def file_to_str(file_path):
    with open(file_path, 'r') as file_buffer:
        return file_buffer.read()


def load_json_config(uri, message):
    return json_format.Parse(file_to_str(uri), message)


def make_dir(path, check_empty=False, force_empty=False, use_dirname=False):
    directory = path
    if use_dirname:
        directory = os.path.dirname(path)

    if force_empty and os.path.isdir(directory):
        shutil.rmtree(directory)

    os.makedirs(directory, exist_ok=True)

    is_empty = len(os.listdir(directory)) == 0
    if check_empty and not is_empty:
        raise ValueError(
            '{} needs to be an empty directory!'.format(directory))


def predict(batch, model):
    # Apply same transform to input as when training.
    # TODO be able to configure this transform and the one in the
    # training generator.
    return model.predict(batch / 255.0)
