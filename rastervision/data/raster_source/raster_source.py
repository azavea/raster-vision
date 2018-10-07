from abc import ABC, abstractmethod


class RasterSource(ABC):
    """A source of raster data.

    This should be subclassed when adding a new source of raster data such as
    a set of files, an API, a TMS URI schema, etc.
    """

    def __init__(self, raster_transformers=[], channel_order=None):
        """Construct a new RasterSource.

        Args:
            raster_transformers: RasterTransformers used to transform chips
                whenever they are retrieved.
            channel_order: numpy array of length n where n is the number of
                channels to use and the values are channel indices.
                Default: None, which will take all the raster's bands as is.

        """
        self.raster_transformers = raster_transformers
        self.channel_order = channel_order

    @abstractmethod
    def get_extent(self):
        """Return the extent of the RasterSource.

        Returns:
            Box in pixel coordinates with extent
        """
        pass

    @abstractmethod
    def get_dtype(self):
        """Return the numpy.dtype of this scene"""
        pass

    @abstractmethod
    def get_crs_transformer(self):
        """Return the associated CRSTransformer."""
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

        if self.channel_order:
            chip = chip[:, :, self.channel_order]

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)

        return chip

    def get_raw_chip(self, window):
        """Return the untransformed chip in the window.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        return self._get_chip(window)

    def get_image_array(self):
        """Return entire transformed image array.

        Not safe to call on very large RasterSources.
        """
        return self.get_chip(self.get_extent())

    def get_raw_image_array(self):
        """Return entire untransformed image array.

        Not safe to call on very large RasterSources.
        """
        return self.get_raw_chip(self.get_extent())
