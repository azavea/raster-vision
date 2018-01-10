from os.path import join, dirname, splitext
import os
from subprocess import Popen
import zipfile
from threading import Timer
from urllib.parse import urlparse

import click

from rv.utils.files import (
    download_if_needed, make_dir, sync_dir, get_local_path,
    MyTemporaryDirectory)
from rv.utils.misc import on_parent_exit
from rv.detection.commands.settings import temp_root_dir


@click.command()
@click.argument('config_uri')
@click.argument('train_dataset_uri')
@click.argument('val_dataset_uri')
@click.argument('model_checkpoint_uri')
@click.argument('train_uri')
@click.option('--sync-interval', default=600,
              help='Interval in seconds for syncing training dir')
@click.option('--save-temp', is_flag=True)
def train(config_uri, train_dataset_uri, val_dataset_uri, model_checkpoint_uri,
          train_uri, sync_interval, save_temp):
    """Train an object detection model.

    Args:
        config_uri: Protobuf file configuring the training
        dataset_uri: Zip file containing dataset
        model_checkpoint_uri: Zip file of pre-trained model checkpoint
        train_uri: Directory for output of training
    """
    prefix = temp_root_dir
    temp_dir = join(prefix, 'train') if save_temp else None
    with MyTemporaryDirectory(temp_dir, prefix) as temp_dir:
        config_path = download_if_needed(config_uri, temp_dir)

        train_root_dir = get_local_path(train_uri, temp_dir)
        make_dir(train_root_dir)
        train_dir = join(train_root_dir, 'train')
        eval_dir = join(train_root_dir, 'eval')

        def process_zip_file(uri, temp_dir, link_dir):
            if uri.endswith('.zip'):
                path = download_if_needed(uri, temp_dir)
                with zipfile.ZipFile(path, 'r') as zip_file:
                    zip_file.extractall(link_dir)
            else:
                make_dir(link_dir, use_dirname=True)
                os.symlink(uri, link_dir)

        train_dataset_dir = join(temp_dir, 'train-dataset')
        process_zip_file(train_dataset_uri, temp_dir, train_dataset_dir)

        val_dataset_dir = join(temp_dir, 'val-dataset')
        process_zip_file(val_dataset_uri, temp_dir, val_dataset_dir)

        model_checkpoint_dir = join(temp_dir, 'model-checkpoint')
        process_zip_file(model_checkpoint_uri, temp_dir, model_checkpoint_dir)

        def sync_train_dir(delete=True):
            sync_dir(train_root_dir, train_uri, delete=delete)
            Timer(sync_interval, sync_train_dir).start()

        if urlparse(train_uri).scheme == 's3':
            sync_train_dir(delete=False)

        train_process = Popen([
            'python', '/opt/src/tf/object_detection/train.py',
            '--logtostderr', '--pipeline_config_path={}'.format(config_path),
            '--train_dir={}'.format(train_dir)],
            preexec_fn=on_parent_exit('SIGTERM'))

        eval_process = Popen([
            'python', '/opt/src/tf/object_detection/eval.py',
            '--logtostderr', '--pipeline_config_path={}'.format(config_path),
            '--checkpoint_dir={}'.format(train_dir),
            '--eval_dir={}'.format(eval_dir)],
            preexec_fn=on_parent_exit('SIGTERM'))

        tensorboard_process = Popen([
            'tensorboard', '--logdir={}'.format(train_root_dir)],
            preexec_fn=on_parent_exit('SIGTERM'))

        train_process.wait()
        eval_process.wait()
        tensorboard_process.wait()


if __name__ == '__main__':
    train()
