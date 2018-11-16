import click

from rastervision.command import Command


class PredictCommand(Command):
    def __init__(self, command_config):
        self.command_config = command_config

    def run(self, tmp_dir=None):
        if not tmp_dir:
            tmp_dir = self.get_tmp_dir()
        msg = 'Making predictions...'

        cc = self.command_config

        backend = cc.backend.create_backend(cc.task)
        task = cc.task.create_task(backend)

        scenes = list(
            map(lambda s: s.create_scene(cc.task, tmp_dir), cc.scenes))

        click.echo(click.style(msg, fg='green'))
        task.predict(scenes, tmp_dir)
