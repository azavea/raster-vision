from rv2.core.command import Command


class Predict(Command):
    def __init__(self, projects, ml_method, raster_transformer, label_map,
                 options):
        self.projects = projects
        self.ml_method = ml_method
        self.raster_transformer = raster_transformer
        self.label_map = label_map
        self.options = options

    def run(self):
        self.ml_method.predict(
            self.projects, self.raster_transformer, self.label_map,
            self.options)
