from rastervision.core.command import Command


class ProcessTrainingData(Command):
    def __init__(self, train_projects, validation_projects, ml_task,
                 options):
        self.train_projects = train_projects
        self.validation_projects = validation_projects
        self.ml_task = ml_task
        self.options = options

    def run(self):
        self.ml_task.process_training_data(
            self.train_projects, self.validation_projects, self.options)
