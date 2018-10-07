from abc import (ABC, abstractmethod)
import os

import rastervision as rv


class RasterSourceDefaultProvider(ABC):
    @staticmethod
    @abstractmethod
    def handles(s):
        """Returns True of this provider is a default for this string"""
        pass

    @abstractmethod
    def construct(s, channel_order=None):
        """Construts a default RasterSource based on the
           string and an optional channel order
        """
        pass


class GeoTiffSourceDefaultProvider(RasterSourceDefaultProvider):
    @staticmethod
    def handles(uri):
        ext = os.path.splitext(uri)[1]
        return ext.lower() in ['.tif', '.tiff']

    @staticmethod
    def construct(uri, channel_order=None):
        return rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                    .with_uri(uri) \
                                    .with_channel_order(channel_order) \
                                    .build()


class GeoJSONSourceDefaultProvider(RasterSourceDefaultProvider):
    @staticmethod
    def handles(uri):
        ext = os.path.splitext(uri)[1]
        return ext.lower() in ['.geojson', '.json']

    @staticmethod
    def construct(uri, channel_order=None):
        return rv.RasterSourceConfig.builder(rv.GEOJSON_SOURCE) \
                                    .with_uri(uri) \
                                    .build()


class ImageSourceDefaultProvider(RasterSourceDefaultProvider):
    @staticmethod
    def handles(uri):
        return True  # This is the catch-all case.

    @staticmethod
    def construct(uri, channel_order=None):
        return rv.RasterSourceConfig.builder(rv.IMAGE_SOURCE) \
                                    .with_uri(uri) \
                                    .with_channel_order(channel_order) \
                                    .build()
