from abc import ABC, abstractmethod


class RasterSource(ABC):
    """A source of raster data.

    This should be subclassed when adding a new source of raster data such as
    a set of files, an API, a TMS URI schema, etc.
    """

    def __init__(self, raster_transformer):
        """Construct a new RasterSource.

        Args:
            raster_transformer: RasterTransformer used to transform chips
                whenever they are retrieved.
        """
        self.raster_transformer = raster_transformer

    @abstractmethod
    def get_extent(self):
        """Return the extent of the RasterSource.

        Returns:
            Box in pixel coordinates with extent
        """
        pass

    @abstractmethod
    def _get_chip(self, window):
        """Return the chip located in the window.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        pass

    def get_chip(self, window):
        """Return the transformed chip in the window.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        chip = self._get_chip(window)
        return self.raster_transformer.transform(chip)

    @abstractmethod
    def get_crs_transformer(self):
        """Return the associated CRSTransformer."""
        pass
