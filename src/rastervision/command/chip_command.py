import click

from rastervision.command import (Command, NoOpCommand)


class ChipCommand(Command):
    def __init__(self, task, augmentors, train_scenes, val_scenes):
        self.task = task
        self.augmentors = augmentors
        self.train_scenes = train_scenes
        self.val_scenes = val_scenes

    def run(self, tmp_dir, dry_run:bool=False):
        msg = 'Making training chips...'
        if dry_run:
            self.announce_dry_run()
        click.echo(click.style(msg, fg='green'))

        if not dry_run:
            self.task.make_chips(self.train_scenes, self.val_scenes,
                                 self.augmentors, tmp_dir)
