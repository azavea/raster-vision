from abc import ABC, abstractmethod

from rv2.core.training_data import TrainingData


class MLTask():
    """Functionality for a specific machine learning task.

    This should be subclassed to add a new task, such as object detection
    """

    def __init__(self, backend, label_map):
        """Construct a new MLTask.

        Args:
            backend: MLBackend
            label_map: LabelMap
        """
        self.backend = backend
        self.label_map = label_map

    @abstractmethod
    def get_train_windows(self, project, options):
        """Return the training windows for a Project.

        The training windows represent the spatial extent of the training
        chips to generate.

        Args:
            project: Project to generate windows for
            options: TrainConfig.Options

        Returns:
            list of Boxes
        """
        pass

    @abstractmethod
    def get_train_annotations(self, window, project, options):
        """Return the training annotations in a window for a project.

        Args:
            window: Box
            project: Project
            options: TrainConfig.Options

        Returns:
            Annotations that lie within window
        """
        pass

    @abstractmethod
    def get_predict_windows(self, extent, options):
        """Return windows to compute predictions for.

        Args:
            extent: Box representing extent of RasterSource
            options: PredictConfig.Options

        Returns:
            list of Boxes
        """
        pass

    @abstractmethod
    def get_evaluation(self):
        """Return empty Evaluation of appropriate type.

        This functions as a factory.
        """
        pass

    def process_training_data(self, train_projects, validation_projects,
                              options):
        """Make training data.

        Convert Projects with a ground_truth_annotation_source into training
        chips in MLBackend-specific format, and write to URI specified in
        options.

        Args:
            train_projects: list of Project
            validation_projects: list of Project
                (that is disjoint from train_projects)
            options: ProcessTrainingDataConfig.Options
        """
        def _process_training_data(projects):
            training_data = TrainingData()
            for project in projects:
                print('Making training chips for project', end='', flush=True)
                windows = self.get_train_windows(project, options)
                for window in windows:
                    chip = project.raster_source.get_chip(window)
                    annotations = self.get_train_annotations(
                        window, project, options)
                    training_data.append(chip, annotations)
                    print('.', end='', flush=True)
                print()
                # TODO load and delete project data as needed to avoid
                # running out of disk space
            return training_data

        training_data = _process_training_data(train_projects)
        validation_data = _process_training_data(validation_projects)
        self.backend.convert_training_data(
            training_data, validation_data, self.label_map, options)

    def train(self, options):
        """Train a model.

        Args:
            options: TrainConfig.options
        """
        self.backend.train(options)

    def predict(self, projects, options):
        """Make predictions for projects.

        The predictions are saved to the prediction_annotation_source in
        each project.

        Args:
            projects: list of Projects
            options: PredictConfig.Options
        """
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
            annotation_source.save(self.label_map)

    def eval(self, projects, options):
        """Evaluate predictions against ground truth in projects.

        Writes output to URI in options.

        Args:
            projects: list of Projects that contain both
                ground_truth_annotation_source and prediction_annotation_source
            options: EvalConfig.Options
        """
        evaluation = self.get_evaluation()
        for project in projects:
            print('Computing evaluation for project...')
            ground_truth = project.ground_truth_annotation_source
            predictions = project.prediction_annotation_source

            project_evaluation = self.get_evaluation()
            project_evaluation.compute(
                self.label_map, ground_truth, predictions)
            evaluation.merge(project_evaluation)
        evaluation.save(options.output_uri)
