from abc import ABC, abstractmethod


class MLBackend(ABC):
    """Functionality for a specific implementation of an MLMethod.

    This should be subclassed to provide a bridge to third party ML libraries.
    """

    @abstractmethod
    def convert_train_data(self, train_data, validation_data, label_map,
                           options):
        """Convert training data to backend-specific format and save it.

        Args:
            train_data: TrainData
            validation_data: TrainData
            label_map: LabelMap
            options: MakeTrainDataConfig.Options
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
        pass
