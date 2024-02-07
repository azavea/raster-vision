from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

if TYPE_CHECKING:
    from rastervision.core.data_sample import DataSample
    from rastervision.core.data import DatasetConfig, Labels, Scene
    from rastervision.core.rv_pipeline import ChipOptions, PredictOptions


class SampleWriter(AbstractContextManager):
    """Writes DataSamples in a streaming fashion.

    This is a context manager used for creating training and validation chips, and
    should be subclassed for each Backend.
    """

    @abstractmethod
    def write_sample(self, sample: 'DataSample'):
        """Writes a single sample."""


class Backend(ABC):
    """Abstraction around core ML functionality used by an RVPipeline.

    This should be subclassed to enable use of a third party ML library with an
    RVPipeline. There is a one-to-many relationship from RVPipeline to Backend.
    """

    @abstractmethod
    def get_sample_writer(self):
        """Returns a SampleWriter for this Backend."""

    @abstractmethod
    def train(self):
        """Train a model.

        This should download chips created by the SampleWriter, train the model, and
        then saving it to disk.
        """

    @abstractmethod
    def load_model(self, uri: Optional[str] = None):
        """Load the model in preparation for one or more prediction calls.

        Args:
            uri: Optional URI to load the model from.
        """

    @abstractmethod
    def predict_scene(self, scene: 'Scene',
                      predict_options: 'PredictOptions') -> 'Labels':
        """Return predictions for an entire scene using the model.

        Args:
            scene: Scene to run inference on.
            predict_options: Prediction options.

        Return:
            Labels object containing predictions
        """

    @abstractmethod
    def chip_dataset(self, dataset: 'DatasetConfig',
                     chip_options: 'ChipOptions') -> None:
        """Create and write chips for scenes in a :class:`.DatasetConfig`.

        Args:
            scenes: Scenes to chip.
        """
