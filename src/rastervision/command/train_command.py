import click

from rastervision.command import Command


class TrainCommand(Command):
    def __init__(self, task):
        self.task = task

    def run(self, tmp_dir, dry_run:bool=False):
        msg = 'Training model...'
        if dry_run:
            self.announce_dry_run()
        click.echo(click.style(msg, fg='green'))
        if not dry_run:
            self.task.train(tmp_dir)
