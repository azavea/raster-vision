from abc import ABC, abstractmethod
from typing import List
from contextlib import AbstractContextManager

import numpy as np

from rastervision.core.data_sample import DataSample
from rastervision.core.box import Box
from rastervision.core.data import Labels


class SampleWriter(AbstractContextManager):
    """Writes DataSamples in a streaming fashion.

    This is a context manager used for creating training and validation chips, and
    should be subclassed for each Backend.
    """

    @abstractmethod
    def write_sample(self, sample: DataSample):
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
    def predict(self, chips: np.ndarray, windows: List[Box]) -> Labels:
        """Return predictions for a batch of chips using the model.

        Args:
            chips: input images of shape [height, width, channels]
            windows: the windows corresponding to the chips in pixel coords

        Return:
            Labels object containing predictions
        """
        pass
