import os
import click

from rastervision.runner import ExperimentRunner
from rastervision.utils.files import save_json_config
from rastervision.cli import Verbosity


def make_command(command_config_uri, tmp_dir=None):
    verbosity = Verbosity.get()
    v_flag = 'v' * max(0, verbosity - 1)
    if v_flag:
        v_flag = '-{}'.format(v_flag)
    if tmp_dir is None:
        command = 'python -m rastervision {} run_command {}'.format(
            v_flag, command_config_uri)
    else:
        command = 'python -m rastervision {} run_command {} --tempdir {}'.format(
            v_flag, command_config_uri, tmp_dir)
    return command


class OutOfProcessExperimentRunner(ExperimentRunner):
    """A class implementing functionality for out-of-process running of experiments.

    The term "out-of-process" refers to the fact that experiments are
    not run within this process, but instead separate processes are
    forked for each stage.

    In the case of `AwsBatchExperimentRunner`, which derives from this
    class, those processes are run remotely on AWS.  In case of
    `LocalExperimentRunner`, which also derives from this class, the
    processes run locally.  This behavior can be contrasted with
    `InProcessExperimentRunner` wherein experiment stages are run
    within the calling process.

    """

    def __init__(self):
        self.tmp_dir = None

    def _run_experiment(self, command_dag):
        """Runs all commands."""

        ids_to_job = {}
        for command_id in command_dag.get_sorted_command_ids():
            command_def = command_dag.get_command_definition(command_id)
            command_config = command_def.command_config
            command_root_uri = command_config.root_uri
            command_basename = 'command-config-{}.json'.format(
                command_config.split_id)
            command_uri = os.path.join(command_root_uri, command_basename)
            print('Saving command configuration to {}...'.format(command_uri))
            save_json_config(command_config.to_proto(), command_uri)

            parent_job_ids = []
            for upstream_id in command_dag.get_upstream_command_ids(
                    command_id):
                if upstream_id not in ids_to_job:
                    cur_command = (command_config.command_type, command_id)
                    u = command_dag.get_command(upstream_id)
                    upstream_command = (u.command_type, upstream_id)
                    raise Exception(
                        '{} command has parent command of {}, '
                        'but does not exist in previous submissions - '
                        'topological sort on command_dag error.'.format(
                            cur_command, upstream_command))
                parent_job_ids.append(ids_to_job[upstream_id])

            run_command = make_command(command_uri, self.tmp_dir)
            job_id = self.submit(
                command_config.command_type, command_config.split_id,
                command_def.experiment_id, run_command, parent_job_ids)

            ids_to_job[command_id] = job_id

    def _dry_run(self, command_dag):
        """Runs all commands."""
        click.echo(
            click.style(
                '\n{} commands to be issued:'.format(
                    self.execution_environment),
                fg='green',
                bold=True,
                underline=True))
        for command_id in command_dag.get_sorted_command_ids():
            command_def = command_dag.get_command_definition(command_id)
            command_config = command_def.command_config
            command_root_uri = command_config.root_uri
            command_uri = os.path.join(command_root_uri, 'command-config.json')
            run_command = make_command(command_uri, self.tmp_dir)
            click.echo('  {}'.format(run_command))
