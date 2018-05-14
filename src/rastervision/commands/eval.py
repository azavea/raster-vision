from rastervision.core.command import Command


class Eval(Command):

    def __init__(self, scenes, ml_task, options):
        self.scenes = scenes
        self.ml_task = ml_task
        self.options = options

    def run(self):
        self.ml_task.eval(self.scenes, self.options)
