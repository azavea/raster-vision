import subprocess

import click

from rastervision.config.utils import (COMPUTE_STATS, MAKE_CHIPS, TRAIN,
                                       PREDICT, EVAL, ALL_COMMANDS)
from rastervision.utils.files import (save_json_config, load_json_config)
from rastervision.protos.experiment_pb2 import ExperimentConfig
from rastervision.utils.batch import _batch_submit
from rastervision import run


def save_command_config(command, config_uri, config):
    save_json_config(config, config_uri)
    print('Wrote {} config to: {}'.format(command, config_uri))


def save_command_configs(experiment, commands):
    if COMPUTE_STATS in commands:
        save_command_config(COMPUTE_STATS, experiment.compute_stats_config_uri,
                            experiment.compute_stats)
    if MAKE_CHIPS in commands:
        save_command_config(MAKE_CHIPS, experiment.make_chips_config_uri,
                            experiment.make_chips)
    if TRAIN in commands:
        save_command_config(TRAIN, experiment.train_config_uri,
                            experiment.train)
    if PREDICT in commands:
        save_command_config(PREDICT, experiment.predict_config_uri,
                            experiment.predict)
    if EVAL in commands:
        save_command_config(EVAL, experiment.eval_config_uri, experiment.eval)


def check_git_branch(branch):
    ls_branch_command = [
        'git', 'ls-remote', '--heads',
        'https://github.com/azavea/raster-vision.git', branch
    ]

    if not subprocess.run(ls_branch_command, stdout=subprocess.PIPE).stdout:
        raise ValueError('Remote branch {} does not exist'.format(branch))


def make_command(command, config_uri):
    return 'python -m rastervision.run {} {}'.format(command, config_uri)


def remote_run(experiment, commands, branch):
    check_git_branch(branch)

    # Run everything in GPU queue since Batch doesn't seem to
    # handle dependencies across different queues.
    parent_job_ids = []

    if COMPUTE_STATS in commands:
        command = make_command(COMPUTE_STATS,
                               experiment.compute_stats_config_uri)
        job_id = _batch_submit(branch, command, attempts=1, gpu=True)
        parent_job_ids = [job_id]

    if MAKE_CHIPS in commands:
        command = make_command(MAKE_CHIPS, experiment.make_chips_config_uri)
        job_id = _batch_submit(
            branch,
            command,
            attempts=1,
            gpu=True,
            parent_job_ids=parent_job_ids)
        parent_job_ids = [job_id]

    if TRAIN in commands:
        command = make_command(TRAIN, experiment.train_config_uri)
        job_id = _batch_submit(
            branch,
            command,
            attempts=1,
            gpu=True,
            parent_job_ids=parent_job_ids)
        parent_job_ids = [job_id]

    if PREDICT in commands:
        command = make_command(PREDICT, experiment.predict_config_uri)
        job_id = _batch_submit(
            branch,
            command,
            attempts=1,
            gpu=True,
            parent_job_ids=parent_job_ids)
        parent_job_ids = [job_id]

    if EVAL in commands:
        command = make_command(EVAL, experiment.eval_config_uri)
        job_id = _batch_submit(
            branch,
            command,
            attempts=1,
            gpu=True,
            parent_job_ids=parent_job_ids)


def local_run(experiment, commands):
    if COMPUTE_STATS in commands:
        run._compute_stats(experiment.compute_stats_config_uri)

    if MAKE_CHIPS in commands:
        run._make_chips(experiment.make_chips_config_uri)

    if TRAIN in commands:
        run._train(experiment.train_config_uri)

    if PREDICT in commands:
        run._predict(experiment.predict_config_uri)

    if EVAL in commands:
        run._eval(experiment.eval_config_uri)


def _main(experiment_uri, commands, remote=False, branch='develop'):
    if len(commands) == 0:
        commands = ALL_COMMANDS

    for command in commands:
        if command not in ALL_COMMANDS:
            raise Exception("Command '{}' is not valid.".format(command))

    experiment = load_json_config(experiment_uri, ExperimentConfig())
    save_command_configs(experiment, commands)

    if remote:
        remote_run(experiment, commands, branch)
    else:
        local_run(experiment, commands)


@click.command()
@click.argument('experiment_uri')
@click.argument('commands', nargs=-1)
@click.option('--remote', is_flag=True)
@click.option('--branch', default='develop')
def main(experiment_uri, commands, remote, branch):
    _main(experiment_uri, commands, remote=remote, branch=branch)


if __name__ == '__main__':
    main()
