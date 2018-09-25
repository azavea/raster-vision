import click

from rastervision.command import Command


class EvalCommand(Command):
    def __init__(self, scenes, evaluators):
        self.scenes = scenes
        self.evaluators = evaluators

    def run(self, tmp_dir, dry_run:bool=False):
        for evaluator in self.evaluators:
            msg = 'Running evaluator: {}...'.format(type(evaluator).__name__)
            if dry_run:
                self.announce_dry_run()
            click.echo(click.style(msg, fg='green'))

            if not dry_run:
                evaluator.process(self.scenes, tmp_dir)
