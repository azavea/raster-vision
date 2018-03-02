from rv2.core.command import Command


class MakeTrainData(Command):
    def __init__(self, train_projects, validation_projects, ml_method,
                 raster_transformer, label_map, options):
        self.train_projects = train_projects
        self.validation_projects = validation_projects
        self.ml_method = ml_method
        self.raster_transformer = raster_transformer
        self.label_map = label_map
        self.options = options

    def run(self):
        self.ml_method.make_train_data(
            self.train_projects, self.validation_projects,
            self.raster_transformer, self.label_map, self.options)
