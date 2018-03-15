from rv2.core.command import Command


class Train(Command):
    def __init__(self, ml_task, options):
        self.ml_task = ml_task
        self.options = options

    def run(self):
        self.ml_task.train(self.options)
