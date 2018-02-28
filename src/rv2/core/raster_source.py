from abc import ABC, abstractmethod


class RasterSource(ABC):
    @abstractmethod
    def get_extent(self):
        pass

    @abstractmethod
    def get_chip(self, window):
        pass

    @abstractmethod
    def get_crs_transformer(self):
        pass
