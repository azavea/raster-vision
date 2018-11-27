import click

from rastervision.command import Command


class TrainCommand(Command):
    def __init__(self, command_config):
        self.command_config = command_config

    def run(self, tmp_dir=None):
        if not tmp_dir:
            tmp_dir = self.get_tmp_dir()
        msg = 'Training model...'
        click.echo(click.style(msg, fg='green'))

        cc = self.command_config

        backend = cc.backend.create_backend(cc.task)
        task = cc.task.create_task(backend)

        task.train(tmp_dir)
