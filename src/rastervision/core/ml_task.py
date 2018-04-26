from abc import ABC, abstractmethod

from rastervision.core.training_data import TrainingData

# TODO: DRY... same keys as in ml_backends/tf_object_detection_aip.py
TRAIN = 'train'
VALIDATION = 'validation'

class MLTask(object):
    """Functionality for a specific machine learning task.

    This should be subclassed to add a new task, such as object detection
    """

    def __init__(self, backend, class_map):
        """Construct a new MLTask.

        Args:
            backend: MLBackend
            class_map: ClassMap
        """
        self.backend = backend
        self.class_map = class_map

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
    def get_train_labels(self, window, project, options):
        """Return the training labels in a window for a project.

        Args:
            window: Box
            project: Project
            options: TrainConfig.Options

        Returns:
            Labels that lie within window
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

    @abstractmethod
    def save_debug_predict_image(self, project, debug_dir_uri):
        """Save a debug image of predictions.

        This writes to debug_dir_uri/<project.id>.jpg.
        """
        pass

    def process_training_data(self, train_projects, validation_projects,
                              options):
        """Process training data.

        Convert Projects with a ground_truth_label_store into training
        chips in MLBackend-specific format, and write to URI specified in
        options.

        Args:
            train_projects: list of Project
            validation_projects: list of Project
                (that is disjoint from train_projects)
            options: ProcessTrainingDataConfig.Options
        """

        def _process_project(project, type_):
            data = TrainingData()
            print('Making {} chips for project: {}'.format(
                type_, project.id), end='', flush=True)
            windows = self.get_train_windows(project, options)
            for window in windows:
                chip = project.raster_source.get_chip(window)
                labels = self.get_train_labels(
                    window, project, options)
                data.append(chip, labels)
                print('.', end='', flush=True)
            print()
            # Shuffle data so the first N samples which are displayed in
            # Tensorboard are more diverse.
            data.shuffle()
            # TODO load and delete project data as needed to avoid
            # running out of disk space
            return self.backend.process_project_data(
                project, data, self.class_map, options)

        def _process_projects(projects, type_):
            return [
                _process_project(project, type_)
                for project in projects
            ]

        # TODO: parallel processing!
        processed_training_results = _process_projects(train_projects, TRAIN)
        processed_validation_results = _process_projects(
            validation_projects, VALIDATION)
        self.backend.process_projectset_results(
            processed_training_results, processed_validation_results,
            self.class_map, options)

    def train(self, options):
        """Train a model.

        Args:
            options: TrainConfig.options
        """
        self.backend.train(self.class_map, options)

    def predict(self, projects, options):
        """Make predictions for projects.

        The predictions are saved to the prediction_label_store in
        each project.

        Args:
            projects: list of Projects
            options: PredictConfig.Options
        """
        for project in projects:
            print('Making predictions for project', end='', flush=True)
            raster_source = project.raster_source
            label_store = project.prediction_label_store
            label_store.clear()

            windows = self.get_predict_windows(
                raster_source.get_extent(), options)
            for window in windows:
                chip = raster_source.get_chip(window)
                labels = self.backend.predict(chip, options)
                label_store.extend(window, labels)
                print('.', end='', flush=True)
            print()

            label_store.post_process(options)
            label_store.save(self.class_map)

            if (options.debug and options.debug_uri and
                    self.class_map.has_all_colors()):
                self.save_debug_predict_image(project, options.debug_uri)

    def eval(self, projects, options):
        """Evaluate predictions against ground truth in projects.

        Writes output to URI in options.

        Args:
            projects: list of Projects that contain both
                ground_truth_label_store and prediction_label_store
            options: EvalConfig.Options
        """
        evaluation = self.get_evaluation()
        for project in projects:
            print('Computing evaluation for project...')
            ground_truth = project.ground_truth_label_store
            predictions = project.prediction_label_store

            project_evaluation = self.get_evaluation()
            project_evaluation.compute(
                self.class_map, ground_truth, predictions)
            evaluation.merge(project_evaluation)
        evaluation.save(options.output_uri)
