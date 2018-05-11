from rastervision.core.command import Command


class Predict(Command):

    def __init__(self, scenes, ml_task, options):
        self.scenes = scenes
        self.ml_task = ml_task
        self.options = options

    def run(self):
        self.ml_task.predict(self.scenes, self.options)
