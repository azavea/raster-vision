import click

from rastervision.command import Command


class AnalyzeCommand(Command):
    def __init__(self, scenes, analyzers):
        self.scenes = scenes
        self.analyzers = analyzers

    def run(self, tmp_dir: str, dry_run:bool=False):
        for analyzer in self.analyzers:
            msg = 'Running analyzer: {}...'.format(type(analyzer).__name__)
            if dry_run:
                self.announce_dry_run()
            click.echo(click.style(msg, fg='green'))
            if not dry_run:
                analyzer.process(self.scenes, tmp_dir)
