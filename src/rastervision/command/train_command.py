import click

from rastervision.command import Command


class TrainCommand(Command):
    def __init__(self, task):
        self.task = task

    def run(self, tmp_dir):
        msg = 'Training model...'
        click.echo(click.style(msg, fg='green'))

        self.task.train(tmp_dir)
