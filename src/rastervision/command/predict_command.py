from rastervision.command import Command


class PredictCommand(Command):
    def __init__(self, task, scenes):
        self.task = task
        self.scenes = scenes

    def run(self, tmp_dir):
        self.task.predict(self.scenes, tmp_dir)
