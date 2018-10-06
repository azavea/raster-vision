import json

from rasterio.features import rasterize
import numpy as np
import shapely

from rastervision.data.raster_source import RasterSource
from rastervision.utils.files import file_to_str
from rastervision.data.utils import geojson_to_shapes


def geojson_to_raster(geojson, extent, crs_transformer):
    # TODO: make this configurable
    line_buffer = 15

    # Crop shapes against extent and remove empty shapes.
    shapes = geojson_to_shapes(geojson, crs_transformer)
    shapes = [s.intersection(extent.to_shapely()) for s in shapes]
    shapes = [s for s in shapes if not s.is_empty]
    shapes = [
        s.buffer(line_buffer) if type(s) is shapely.geometry.LineString else s
        for s in shapes
    ]

    # TODO: make this configurable
    # Map background to class 1 and shapes to class 2.
    shape_vals = [(shape, 2) for shape in shapes]
    out_shape = (extent.get_height(), extent.get_width())
    if shapes:
        raster = rasterize(shape_vals, out_shape=out_shape, fill=1)
    else:
        raster = np.ones(out_shape)

    return raster


class GeoJSONSource(RasterSource):
    def __init__(self, uri, extent, crs_transformer):
        self.uri = uri
        self.extent = extent
        self.crs_transformer = crs_transformer
        geojson_dict = json.loads(file_to_str(self.uri))
        self.raster = geojson_to_raster(geojson_dict, extent, crs_transformer)
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
