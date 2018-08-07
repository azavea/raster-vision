from rastervision.core.command import Command


class MakeTrainingChips(Command):
    def __init__(self, train_scenes, validation_scenes, ml_task, options):
        self.train_scenes = train_scenes
        self.validation_scenes = validation_scenes
        self.ml_task = ml_task
        self.options = options

    def run(self):
        self.ml_task.make_training_chips(self.train_scenes,
                                         self.validation_scenes, self.options)
