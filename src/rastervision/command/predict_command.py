import click

from rastervision.command import (Command, NoOpCommand)


class PredictCommand(Command):
    def __init__(self, task, scenes):
        self.task = task
        self.scenes = scenes

    def run(self, tmp_dir):
        msg = 'Making predictions...'
        click.echo(click.style(msg, fg='green'))
        self.task.predict(self.scenes, tmp_dir)

class NoOpPredictCommand(NoOpCommand):
    def __init__(self, task, scenes):
        self.task = task
        self.scenes = scenes

    def run(self, tmp_dir):
        self.announce()
        msg = 'Making predictions...'
        click.echo(click.style(msg, fg='green'))
