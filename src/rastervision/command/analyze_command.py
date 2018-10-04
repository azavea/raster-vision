import click

from rastervision.command import Command


class AnalyzeCommand(Command):
    def __init__(self, scenes, analyzers):
        self.scenes = scenes
        self.analyzers = analyzers

    def run(self, tmp_dir=None):
        if not tmp_dir:
            tmp_dir = self.get_tmp_dir()
        for analyzer in self.analyzers:
            msg = 'Running analyzer: {}...'.format(type(analyzer).__name__)
            click.echo(click.style(msg, fg='green'))

            analyzer.process(self.scenes, tmp_dir)
