from os.path import join, dirname
from os import makedirs
from subprocess import Popen
import zipfile
from threading import Timer
from urllib.parse import urlparse

import click

from rv.commands.utils import (
    download_if_needed, make_temp_dir, on_parent_exit, sync_dir)


@click.command()
@click.argument('config_path')
@click.argument('dataset_uri')
@click.argument('model_checkpoint_uri')
@click.argument('train_uri')
@click.option('--sync-interval', default=600,
              help='Interval in seconds for syncing training dir')
def train(config_path, dataset_uri, model_checkpoint_uri, train_uri,
          sync_interval):
    """Train an object detection model.

    Args:
        train_uri: Directory for output of training
        dataset_uri: Zip file containing dataset
        model_checkpoint_uri: Zip file of pre-trained model checkpoint
    """
    temp_dir = '/opt/data/temp/'
    download_dir = '/opt/data/'
    make_temp_dir(temp_dir)

    train_root_dir = download_if_needed(
        temp_dir, train_uri, must_exist=False)
    makedirs(train_root_dir, exist_ok=True)
    train_dir = join(train_root_dir, 'train')
    eval_dir = join(train_root_dir, 'eval')

    dataset_path = download_if_needed(download_dir, dataset_uri)
    with zipfile.ZipFile(dataset_path, 'r') as dataset_file:
        dataset_file.extractall(dirname(dataset_path))
    model_checkpoint_path = download_if_needed(
        download_dir, model_checkpoint_uri)
    with zipfile.ZipFile(model_checkpoint_path, 'r') as model_checkpoint_file:
        model_checkpoint_file.extractall(dirname(model_checkpoint_path))

    def sync_train_dir():
        sync_dir(train_root_dir, train_uri)
        Timer(sync_interval, sync_train_dir).start()

    if urlparse(train_uri).scheme == 's3':
        sync_train_dir()

    train_process = Popen([
        'python', '/opt/src/detection/models/object_detection/train.py',
        '--logtostderr', '--pipeline_config_path={}'.format(config_path),
        '--train_dir={}'.format(train_dir)],
        preexec_fn=on_parent_exit('SIGTERM'))

    eval_process = Popen([
        'python', '/opt/src/detection/models/object_detection/eval.py',
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
