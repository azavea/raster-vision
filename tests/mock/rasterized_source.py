from unittest.mock import Mock
import json

from rastervision.data import (RasterizedSource, RasterizedSourceConfig,
                               VectorSource, IdentityCRSTransformer)
from rastervision.core import Box


class MockRasterizedSource(RasterizedSource):
    class MockVectorSource(VectorSource):
        def __init__(self, uri):
            self.mock = Mock()
            with open(uri) as f:
                self.geojson = json.load(f)

        def _get_geojson(self):
            return ''

    def __init__(self, uri, class_id=0):
        self.mock = Mock()
        self.vector_source = MockRasterizedSource.MockVectorSource(uri)
        self.crs_transformer = IdentityCRSTransformer()
        self.extent = Box(0, 0, 360, 360)
        self.rasterizer_options = RasterizedSourceConfig.RasterizerOptions(
            0xff)

    def _get_geojson(self):
        self.vector_source.get_geojson()
