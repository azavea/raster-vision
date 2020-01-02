from abc import (ABC, abstractmethod)

import rastervision as rv


class RasterSourceDefaultProvider(ABC):
    @staticmethod
    @abstractmethod
    def handles(s):
        """Returns True if this provider is a default for this string"""
        pass

    @abstractmethod
    def construct(s, channel_order=None):
        """Constructs default based on the string and an optional channel order."""
        pass


class RasterioSourceDefaultProvider(RasterSourceDefaultProvider):
    @staticmethod
    def handles(uri):
        # Since there are so many types handled by Rasterio/GDAL, the RasterioSource
        # will be the catch-all. More specific types can be handled by other
        # RasterSources.
        return True

    @staticmethod
    def construct(uri, channel_order=None):
        return rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                                    .with_uri(uri) \
                                    .with_channel_order(channel_order) \
                                    .build()
