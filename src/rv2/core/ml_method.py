from abc import ABC, abstractmethod

from rv2.core.train_data import TrainData


# TODO Rename to MLTask
class MLMethod():
    """An abstract machine learning method (eg. object detection)."""
    def __init__(self, backend):
        self.backend = backend

    @abstractmethod
    def get_train_windows(self, raster_source, annotation_source, options):
        pass

    @abstractmethod
    def get_train_annotations(self, window, raster_source, annotation_source,
                              options):
        pass

    @abstractmethod
    def get_predict_windows(self, extent, options):
        pass

    def make_train_data(self, train_projects, validation_projects, label_map,
                        options):
        def _make_train_data(projects):
            train_data = TrainData()
            for project in projects:
                print('Making training chips for project', end='', flush=True)

                raster_source = project.raster_source
                annotation_source = project.ground_truth_annotation_source
                # Each window is an extent, not the actual image for that extent.
                windows = self.get_train_windows(
                    raster_source, annotation_source, options)

                for window in windows:
                    chip = raster_source.get_chip(window)
                    annotations = self.get_train_annotations(
                        window, raster_source, annotation_source, options)
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

    def predict(self, projects, label_map, options):
        for project in projects:
            print('Making predictions for project', end='', flush=True)
            raster_source = project.raster_source
            annotation_source = project.prediction_annotation_source
            annotation_source.clear()

            windows = self.get_predict_windows(
                raster_source.get_extent(), options)
            for window in windows:
                chip = raster_source.get_chip(window)
                annotations = self.backend.predict(chip, options)
                annotation_source.extend(window, annotations)
                print('.', end='', flush=True)
            print()

            annotation_source.post_process(options)
            annotation_source.save(label_map)

    @abstractmethod
    def get_evaluation(self):
        pass

    def eval(self, projects, label_map, options):
        evaluation = self.get_evaluation()
        for project in projects:
            print('Computing evaluation for project...')
            ground_truth = project.ground_truth_annotation_source
            predictions = project.prediction_annotation_source

            project_evaluation = self.get_evaluation()
            project_evaluation.compute(label_map, ground_truth, predictions)
            evaluation.merge(project_evaluation)
        evaluation.save(options.output_uri)
