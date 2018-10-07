import click

from rastervision.command import Command


class ChipCommand(Command):
    def __init__(self, task, augmentors, train_scenes, val_scenes):
        self.task = task
        self.augmentors = augmentors
        self.train_scenes = train_scenes
        self.val_scenes = val_scenes

    def run(self, tmp_dir=None):
        if not tmp_dir:
            tmp_dir = self.get_tmp_dir()
        msg = 'Making training chips...'
        click.echo(click.style(msg, fg='green'))

        self.task.make_chips(self.train_scenes, self.val_scenes,
                             self.augmentors, tmp_dir)
