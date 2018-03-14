from abc import ABC, abstractmethod


class RasterSource(ABC):
    """A source of raster data.

    A RasterSource is a source of raster data that can be implemented by a
    set of files, an API, a TMS URI schema, etc.
    """
    def __init__(self, raster_transformer):
        self.raster_transformer = raster_transformer

    @abstractmethod
    def get_extent(self):
        pass

    @abstractmethod
    def _get_chip(self, window):
        pass

    def get_chip(self, window):
        chip = self._get_chip(window)
        return self.raster_transformer.transform(chip)

    @abstractmethod
    def get_crs_transformer(self):
        pass
