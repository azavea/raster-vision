import click

from rastervision.command import Command


class ChipCommand(Command):
    def __init__(self, command_config):
        self.command_config = command_config

    def run(self, tmp_dir=None, index: int = 0, count: int = 1):
        if not tmp_dir:
            tmp_dir = self.get_tmp_dir()
        msg = 'Making training chips...'
        click.echo(click.style(msg, fg='green'))

        cc = self.command_config

        backend = cc.backend.create_backend(cc.task, index=index, count=count)
        task = cc.task.create_task(backend)

        train_scenes = list(
            map(lambda s: s.create_scene(cc.task, tmp_dir),
                cc.train_scenes[index::count]))

        val_scenes = list(
            map(lambda s: s.create_scene(cc.task, tmp_dir),
                cc.val_scenes[index::count]))

        augmentors = list(map(lambda a: a.create_augmentor(), cc.augmentors))

        task.make_chips(train_scenes, val_scenes, augmentors, tmp_dir)
