from os.path import join, splitext
from os import makedirs
import zipfile
from threading import Timer
from urllib.parse import urlparse

import click

from rv.utils import (
    download_if_needed, make_empty_dir, sync_dir, get_local_path)
from rv.classification.commands.settings import temp_root_dir
from rv.classification.ml.train import _train as train_model


def get_dataset(download_dir, dataset_uri):
    dataset_path = download_if_needed(download_dir, dataset_uri)
    with zipfile.ZipFile(dataset_path, 'r') as dataset_file:
        dataset_dir = splitext(dataset_path)[0]
        dataset_file.extractall(dataset_dir)
    return dataset_dir


@click.command()
@click.argument('config_uri')
@click.argument('train_dataset_uri')
@click.argument('val_dataset_uri')
@click.argument('output_uri')
@click.option('--sync-interval', default=600,
              help='Interval in seconds for syncing training dir')
def train(config_uri, train_dataset_uri, val_dataset_uri, output_uri,
          sync_interval):
    """Train a classification model.

    Args:
    """
    temp_dir = join(temp_root_dir, 'train')
    download_dir = '/opt/data/'
    make_empty_dir(temp_dir)

    output_dir = download_if_needed(temp_dir, output_uri, must_exist=False)
    makedirs(output_dir, exist_ok=True)

    config_path = download_if_needed(temp_dir, config_uri, must_exist=True)
    train_dataset_dir = get_dataset(download_dir, train_dataset_uri)
    val_dataset_dir = get_dataset(download_dir, val_dataset_uri)

    def sync_train_dir(delete=True):
        sync_dir(output_dir, output_uri, delete=delete)
        Timer(sync_interval, sync_train_dir).start()

    if urlparse(output_uri).scheme == 's3':
        # Download anything saved from previous run.
        sync_dir(output_uri, output_dir, delete=False)
        # Start periodically uploading to S3.
        sync_train_dir(delete=False)

    train_model(config_path, train_dataset_dir, val_dataset_dir, output_dir)
    # Upload final results to S3.
    sync_dir(output_dir, output_uri, delete=True)


if __name__ == '__main__':
    train()
