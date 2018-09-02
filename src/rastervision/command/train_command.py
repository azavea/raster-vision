from rastervision.command import Command

class TrainCommand(Command):
    def __init__(self, task):
        self.task = task

    def run(self, tmp_dir):
        self.task.train(tmp_dir)
