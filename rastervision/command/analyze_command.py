import click

from rastervision.command import Command


class AnalyzeCommand(Command):
    def __init__(self, command_config):
        self.command_config = command_config

    def run(self, tmp_dir=None):
        if not tmp_dir:
            tmp_dir = self.get_tmp_dir()

        cc = self.command_config

        analyzers = list(map(lambda a: a.create_analyzer(), cc.analyzers))
        scenes = list(
            map(lambda s: s.create_scene(cc.task, tmp_dir), cc.scenes))

        for analyzer in analyzers:
            msg = 'Running analyzer: {}...'.format(type(analyzer).__name__)
            click.echo(click.style(msg, fg='green'))

            analyzer.process(scenes, tmp_dir)
