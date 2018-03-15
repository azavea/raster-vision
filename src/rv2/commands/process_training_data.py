from rv2.core.command import Command


class ProcessTrainingData(Command):
    def __init__(self, train_projects, validation_projects, ml_task,
                 label_map, options):
        self.train_projects = train_projects
        self.validation_projects = validation_projects
        self.ml_task = ml_task
        self.label_map = label_map
        self.options = options

    def run(self):
        self.ml_task.process_training_data(
            self.train_projects, self.validation_projects,
            self.label_map, self.options)
