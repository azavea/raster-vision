import click

from rastervision.command import Command


class EvalCommand(Command):
    def __init__(self, command_config):
        self.command_config = command_config

    def run(self, tmp_dir=None):
        if not tmp_dir:
            tmp_dir = self.get_tmp_dir()

        cc = self.command_config

        scenes = list(
            map(lambda s: s.create_scene(cc.task, tmp_dir), cc.scenes))
        evaluators = list(map(lambda a: a.create_evaluator(), cc.evaluators))

        for evaluator in evaluators:
            msg = 'Running evaluator: {}...'.format(type(evaluator).__name__)
            click.echo(click.style(msg, fg='green'))

            evaluator.process(scenes, tmp_dir)
