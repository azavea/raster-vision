import click

from rastervision.command import (Command, NoOpCommand)


class EvalCommand(Command):
    def __init__(self, scenes, evaluators):
        self.scenes = scenes
        self.evaluators = evaluators

    def run(self, tmp_dir):
        for evaluator in self.evaluators:
            msg = 'Running evaluator: {}...'.format(type(evaluator).__name__)
            click.echo(click.style(msg, fg='green'))

            evaluator.process(self.scenes, tmp_dir)

class NoOpEvalCommand(NoOpCommand):
    def __init__(self, scenes, evaluators):
        self.scenes = scenes
        self.evaluators = evaluators

    def run(self, tmp_dir):
        for evaluator in self.evaluators:
            self.announce()
            msg = 'Running evaluator: {}...'.format(type(evaluator).__name__)
            click.echo(click.style(msg, fg='green'))
