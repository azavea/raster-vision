import json

from rasterio.features import rasterize
import numpy as np
import shapely

from rastervision.data.raster_source import RasterSource
from rastervision.utils.files import file_to_str
from rastervision.data.utils import geojson_to_shapes


def geojson_to_raster(geojson, rasterizer_options, extent, crs_transformer):
    line_buffer = rasterizer_options.line_buffer
    background_class_id = rasterizer_options.background_class_id

    # Crop shapes against extent and remove empty shapes.
    shapes = geojson_to_shapes(geojson, crs_transformer)
    shapes = [(s.intersection(extent.to_shapely()), c) for s, c in shapes]
    shapes = [(s, c) for s, c in shapes if not s.is_empty]
    shapes = [(s.buffer(line_buffer), c)
              if type(s) is shapely.geometry.LineString else (s, c)
              for s, c in shapes]

    out_shape = (extent.get_height(), extent.get_width())
    # rasterize needs to passed >= 1 shapes.
    if shapes:
        raster = rasterize(
            shapes, out_shape=out_shape, fill=background_class_id)
    else:
        raster = np.full(out_shape, background_class_id)

    return raster


class GeoJSONSource(RasterSource):
    """A RasterSource based on the rasterization of a GeoJSON file."""

    def __init__(self, uri, rasterizer_options, extent, crs_transformer):
        """Constructor.

        Args:
            uri: URI of GeoJSON file
            rasterizer_options:
                rastervision.data.raster_source.GeoJSONSourceConfig.RasterizerOptions
            extent: (Box) extent of corresponding imagery RasterSource
            crs_transformer: (CRSTransformer)
        """
        self.uri = uri
        self.rasterizer_options = rasterizer_options
        self.extent = extent
        self.crs_transformer = crs_transformer
        geojson = json.loads(file_to_str(self.uri))
        self.raster = geojson_to_raster(geojson, rasterizer_options, extent,
                                        crs_transformer)
        # Add third singleton dim since rasters must have >=1 channel.
        self.raster = np.expand_dims(self.raster, 2)
        super().__init__()

    def get_extent(self):
        """Return the extent of the RasterSource.

        Returns:
            Box in pixel coordinates with extent
        """
        return self.extent

    def get_dtype(self):
        """Return the numpy.dtype of this scene"""
        return np.uint8

    def get_crs_transformer(self):
        """Return the associated CRSTransformer."""
        return self.crs_transformer

    def _get_chip(self, window):
        """Return the chip located in the window.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        return self.raster[window.ymin:window.ymax, window.xmin:window.xmax, :]
