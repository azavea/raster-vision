from abc import ABC, abstractmethod

from rastervision.v2.rv.data_sample import DataSample


class SampleWriter(ABC):
    @abstractmethod
    def write_sample(sample: DataSample):
        pass


class Backend(ABC):
    """Functionality for a specific implementation of an MLTask.

    This should be subclassed to provide a bridge to third party ML libraries.
    There is a many-to-one relationship from backends to tasks.
    """

    @abstractmethod
    def get_sample_writer(self):
        pass

    @abstractmethod
    def train(self):
        """Train a model.
        """
        pass

    @abstractmethod
    def load_model(self):
        """Load the model in preparation for one or more prediction calls."""
        pass

    @abstractmethod
    def predict(self, chips, windows):
        """Return predictions for a chip using model.

        Args:
            chips: [[height, width, channels], ...] numpy array of chips
            windows: List of boxes that are the windows aligned with the chips.

        Return:
            Labels object containing predictions
        """
        pass
