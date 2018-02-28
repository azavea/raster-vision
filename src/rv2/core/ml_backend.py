from abc import ABC, abstractmethod


class MLBackend(ABC):
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
