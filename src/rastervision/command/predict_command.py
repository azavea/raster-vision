import click

from rastervision.command import Command


class PredictCommand(Command):
    def __init__(self, task, scenes):
        self.task = task
        self.scenes = scenes

    def run(self, tmp_dir, dry_run: bool = False):
        msg = 'Making predictions...'
        if dry_run:
            self.announce_dry_run()
        click.echo(click.style(msg, fg='green'))
        if not dry_run:
            self.task.predict(self.scenes, tmp_dir)
