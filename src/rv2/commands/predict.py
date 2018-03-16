from rv2.core.command import Command


class Predict(Command):
    def __init__(self, projects, ml_task, options):
        self.projects = projects
        self.ml_task = ml_task
        self.options = options

    def run(self):
        self.ml_task.predict(self.projects, self.options)
