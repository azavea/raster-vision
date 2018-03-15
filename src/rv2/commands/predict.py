from rv2.core.command import Command


class Predict(Command):
    def __init__(self, projects, ml_task, label_map, options):
        self.projects = projects
        self.ml_task = ml_task
        self.label_map = label_map
        self.options = options

    def run(self):
        self.ml_task.predict(self.projects, self.label_map, self.options)
