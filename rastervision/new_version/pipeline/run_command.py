import os
import tempfile

import click

from rastervision.utils.files import file_to_json, make_dir
from rastervision.new_version.pipeline.config import build_config


def _run_command(cfg_json_uri, command, split_ind, num_splits):
    tmp_root_dir = '/opt/data/tmp'
    make_dir(tmp_root_dir)
    tmp_dir_obj = tempfile.TemporaryDirectory(dir=tmp_root_dir)
    tmp_dir = tmp_dir_obj.name

    pipeline_cfg_dict = file_to_json(cfg_json_uri)
    cfg = build_config(pipeline_cfg_dict)
    pipeline = cfg.get_pipeline()(cfg, tmp_dir)

    # TODO generalize this to work outside batch
    if split_ind is None:
        split_ind = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))
    command_fn = getattr(pipeline, command)

    if num_splits > 1:
        print('Running {} command split {}/{}...'.format(
            command, split_ind + 1, num_splits))
        command_fn(split_ind=split_ind, num_splits=num_splits)
    else:
        print('Running {} command...'.format(command))
        command_fn()


@click.command()
@click.argument('cfg_json_uri')
@click.argument('command')
@click.option('--split-ind')
@click.option('--num-splits', default=1)
def run_command(cfg_json_uri, command, split_ind, num_splits):
    _run_command(cfg_json_uri, command, split_ind, num_splits)


if __name__ == '__main__':
    run_command()
