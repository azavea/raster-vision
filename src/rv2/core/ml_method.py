from abc import ABC, abstractmethod

from rv2.core.train_data import TrainData


class MLMethod():
    def __init__(self, backend):
        self.backend = backend

    @abstractmethod
    def get_train_windows(self, extent, annotation_source, options):
        pass

    @abstractmethod
    def get_train_annotations(self, window, annotation_source, options):
        pass

    def make_train_data(self, train_projects, validation_projects,
                        raster_transformer, label_map, options):
        def _make_train_data(projects):
            train_data = TrainData()
            for project in projects:
                print('Making training chips for project', end='', flush=True)
                raster_source = project.raster_source
                annotation_source = project.annotation_source

                # Each window is an extent, not the actual image for that extent.
                windows = self.get_train_windows(
                    raster_source.get_extent(), annotation_source, options)
                for window in windows:
                    chip = raster_source.get_chip(window)
                    chip = raster_transformer.transform(chip)
                    annotations = self.get_train_annotations(
                        window, annotation_source, options)
                    train_data.append(chip, annotations)
                    print('.', end='', flush=True)
                print()
            return train_data

        train_data = _make_train_data(train_projects)
        validation_data = _make_train_data(validation_projects)
        self.backend.convert_train_data(
            train_data, validation_data, label_map, options)

    def train(self, options):
        self.backend.train(options)

    @abstractmethod
    def get_predict_windows(self, extent, options):
        pass

    def predict(self, projects, raster_transformer, label_map, options):
        for project in projects:
            print('Making predictions for project', end='', flush=True)
            raster_source = project.raster_source
            annotation_source = project.annotation_source

            windows = self.get_predict_windows(
                raster_source.get_extent(), options)
            for window in windows:
                chip = raster_source.get_chip(window)
                chip = raster_transformer.transform(chip)
                annotations = self.backend.predict(chip, options)
                annotation_source.extend(window, annotations)
                print('.', end='', flush=True)
            print()

            annotation_source.post_process(options)
            annotation_source.save(label_map)
