from abc import ABC, abstractmethod


class MLBackend(ABC):
    """Bridge to a concrete implementation of an MLMethod.

    This should be subclassed to provide a unified interface to the
    functionality needed by RV to third party ML libraries.
    """
    @abstractmethod
    def convert_train_data(self, train_data, validation_data, label_map,
                           options):
        pass

    @abstractmethod
    def train(self, options):
        pass

    @abstractmethod
    def predict(self, chip, options):
        pass
