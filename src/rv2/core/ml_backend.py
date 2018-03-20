from abc import ABC, abstractmethod


class MLBackend(ABC):
    """Functionality for a specific implementation of an MLTask.

    This should be subclassed to provide a bridge to third party ML libraries.
    """

    @abstractmethod
    def convert_training_data(self, training_data, validation_data, class_map,
                              options):
        """Convert training data to backend-specific format and save it.

        Args:
            training_data: TrainingData
            validation_data: TrainingData
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
