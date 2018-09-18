import click

from rastervision.command import Command


class PredictCommand(Command):
    def __init__(self, task, scenes):
        self.task = task
        self.scenes = scenes

    def run(self, tmp_dir):
        msg = 'Making predictions...'
        click.echo(click.style(msg, fg='green'))
        self.task.predict(self.scenes, tmp_dir)
