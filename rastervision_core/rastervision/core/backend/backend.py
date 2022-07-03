from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from contextlib import AbstractContextManager

if TYPE_CHECKING:
    from rastervision.core.data_sample import DataSample
    from rastervision.core.data import Labels, Scene


class SampleWriter(AbstractContextManager):
    """Writes DataSamples in a streaming fashion.

    This is a context manager used for creating training and validation chips, and
    should be subclassed for each Backend.
    """

    @abstractmethod
    def write_sample(self, sample: 'DataSample'):
        """Writes a single sample."""
        pass


class Backend(ABC):
    """Abstraction around core ML functionality used by an RVPipeline.

    This should be subclassed to enable use of a third party ML library with an
    RVPipeline. There is a one-to-many relationship from RVPipeline to Backend.
    """

    @abstractmethod
    def get_sample_writer(self):
        """Returns a SampleWriter for this Backend."""
        pass

    @abstractmethod
    def train(self):
        """Train a model.

        This should download chips created by the SampleWriter, train the model, and
        then saving it to disk.
        """
        pass

    @abstractmethod
    def load_model(self):
        """Load the model in preparation for one or more prediction calls."""
        pass

    @abstractmethod
    def predict_scene(self, scene: 'Scene', chip_sz: int,
                      stride: int) -> 'Labels':
        """Return predictions for an entire scene using the model.

        Args:
            scene (Scene): Scene to run inference on.

        Return:
            Labels object containing predictions
        """
        pass
