import logging

from rasterio.features import rasterize
import numpy as np
import shapely
from shapely.strtree import STRtree

from rastervision.data import (ActivateMixin, ActivationError)
from rastervision.data.raster_source import RasterSource
from rastervision.data.utils import geojson_to_shapes

log = logging.getLogger(__name__)


def geojson_to_raster(str_tree, rasterizer_options, extent, crs_transformer):
    line_buffer = rasterizer_options.line_buffer
    background_class_id = rasterizer_options.background_class_id

    log.debug('Cropping shapes to chip extent...')
    # Crop shapes against extent, remove empty shapes, and put in extent frame of
    # reference.
    shapes = str_tree.query(extent.to_shapely())
    shapes = [(s, s.class_id) for s in shapes]
    shapes = [(s.intersection(extent.to_shapely()), c) for s, c in shapes]
    shapes = [(s, c) for s, c in shapes if not s.is_empty]
    shapes = [(s.buffer(line_buffer), c)
              if type(s) is shapely.geometry.LineString else (s, c)
              for s, c in shapes]
    shapes = [(shapely.ops.transform(lambda x, y, z=None: (x - extent.xmin, y - extent.ymin), s),
               c)
              for s, c in shapes]
    log.debug('# of shapes in extent: {}'.format(len(shapes)))

    out_shape = (extent.get_height(), extent.get_width())
    # rasterize needs to be passed >= 1 shapes.
    if shapes:
        log.debug('rasterio.rasterize()...')
        raster = rasterize(
            shapes, out_shape=out_shape, fill=background_class_id)
    else:
        raster = np.full(out_shape, background_class_id)

    return raster


class RasterizedSource(ActivateMixin, RasterSource):
    """A RasterSource based on the rasterization of a VectorSource."""

    def __init__(self, vector_source, rasterizer_options, extent,
                 crs_transformer):
        """Constructor.

        Args:
            vector_source: (VectorSource)
            rasterizer_options:
                rastervision.data.raster_source.GeoJSONSourceConfig.RasterizerOptions
            extent: (Box) extent of corresponding imagery RasterSource
            crs_transformer: (CRSTransformer)
        """
        self.vector_source = vector_source
        self.rasterizer_options = rasterizer_options
        self.extent = extent
        self.crs_transformer = crs_transformer
        self.activated = False

        super().__init__(channel_order=[0], num_channels=1)

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
        if not self.activated:
            raise ActivationError('GeoJSONSource must be activated before use')

        log.debug('Rasterizing window: {}'.format(window))
        chip = geojson_to_raster(self.str_tree, self.rasterizer_options,
                                 window, self.crs_transformer)
        # Add third singleton dim since rasters must have >=1 channel.
        return np.expand_dims(chip, 2)

    def _activate(self):
        geojson = self.vector_source.get_geojson()
        shapes = geojson_to_shapes(geojson, self.crs_transformer)

        # Monkey-patching class_id onto shapely.geom is not a good idea because
        # if you transform it, the class_id will be lost, but this works here. I wanted to
        # use a dictionary to associate shape with class_id, but couldn't because they are
        # mutable.
        for shape, class_id in shapes:
            shape.class_id = class_id
        self.str_tree = STRtree([shape for shape, class_id in shapes])
        self.activated = True

    def _deactivate(self):
        self.str_tree = None
        self.activated = False
