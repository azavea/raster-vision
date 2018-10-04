from abc import (ABC, abstractmethod)


class RasterTransformer(ABC):
    """Transforms raw chips to be input to a neural network."""

    @abstractmethod
    def transform(self, chip, channel_order=None):
        """Transform a chip of a raster source.

        Args:
            chip: [height, width, channels] numpy array

        Returns:
            [height, width, channels] numpy array

        """
        pass
