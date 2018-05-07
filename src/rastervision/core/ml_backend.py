from abc import ABC, abstractmethod


class MLBackend(ABC):
    """Functionality for a specific implementation of an MLTask.

    This should be subclassed to provide a bridge to third party ML libraries.
    There is a many-to-one relationship from backends to tasks.
    """

    @abstractmethod
    def process_project_data(self, project, data, class_map, options):
        """Process each project's training data

        Args:
            project: Project
            data: TrainingData
            class_map: ClassMap
            options: ProcessTrainingDataConfig.Options

        Returns:
            backend-specific data-structures consumed by ml_backend's
            process_projectset_results
        """
        pass

    @abstractmethod
    def process_projectset_results(self, training_results, validation_results,
                                       class_map, options):
        """After all projects have been processed, process the resultset

        Args:
            training_results: dependent on the ml_backend's process_project_data
            validation_results: dependent on the ml_backend's process_project_data
            class_map: ClassMap
            options: ProcessTrainingDataConfig.Options
        """
        pass

    @abstractmethod
    def train(self, options):
        """Train a model.

        Args:
            options: TrainConfig.Options
        """
        pass

    @abstractmethod
    def predict(self, chip, options):
        """Return predictions for a chip using model.

        Args:
            chip: [height, width, channels] numpy array
            options: PredictConfig.Options
        """
        # TODO predict by the batch-load to make better use of the gpu
        pass
