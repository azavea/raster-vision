from typing import TYPE_CHECKING, List
import logging

from rasterio.features import rasterize
import numpy as np
from shapely.geometry import shape
from shapely.strtree import STRtree
from shapely.ops import transform

from rastervision.core.data.raster_source import RasterSource

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rastervision.core.box import Box
    from rastervision.core.data import VectorSource, RasterTransformer


def geoms_to_raster(str_tree: STRtree, window: 'Box', background_class_id: int,
                    all_touched: bool, extent: 'Box') -> np.ndarray:
    # Crop shapes against window, remove empty shapes, and put in window frame
    # of reference.
    log.debug('Cropping shapes to window...')
    window_geom = window.to_shapely()
    shapes = str_tree.query(window_geom)
    shapes = [(s, s.class_id) for s in shapes]
    shapes = [(s.intersection(window_geom), c) for s, c in shapes]
    shapes = [(s, c) for s, c in shapes if not s.is_empty]

    def to_window_frame(x, y, z=None):
        return (x - window.xmin, y - window.ymin)

    shapes = [(transform(to_window_frame, s), c) for s, c in shapes]
    log.debug('# of shapes in window: {}'.format(len(shapes)))

    out_shape = (window.get_height(), window.get_width())

    # rasterize needs to be passed >= 1 shapes.
    if shapes:
        log.debug('rasterio.rasterize()...')
        raster = rasterize(
            shapes,
            out_shape=out_shape,
            fill=background_class_id,
            dtype=np.uint8,
            all_touched=all_touched)
    else:
        raster = np.full(out_shape, background_class_id, dtype=np.uint8)

    return raster


class RasterizedSource(RasterSource):
    """A RasterSource based on the rasterization of a VectorSource."""

    def __init__(self,
                 vector_source: 'VectorSource',
                 background_class_id: int,
                 extent: 'Box',
                 all_touched: bool = False,
                 raster_transformers: List['RasterTransformer'] = []):
        """Constructor.

        Args:
            vector_source (VectorSource): The VectorSource to rasterize.
            background_class_id (int): The class_id to use for any background
                pixels, ie. pixels not covered by a polygon.
            extent (Box): Extent of corresponding imagery RasterSource.
            all_touched (bool, optional): If True, all pixels touched by
                geometries will be burned in. If false, only pixels whose
                center is within the polygon or that are selected by
                Bresenham's line algorithm will be burned in.
                (See rasterio.features.rasterize for more details). Defaults
                to False.
        """
        self.vector_source = vector_source
        self.background_class_id = background_class_id
        self.extent = extent
        self.all_touched = all_touched
        self.str_tree = self.make_str_tree()

        super().__init__(
            channel_order=[0],
            num_channels_raw=1,
            raster_transformers=raster_transformers)

    def get_extent(self):
        """Return the extent of the RasterSource.

        Returns:
            Box in pixel coordinates with extent
        """
        return self.extent

    @property
    def dtype(self) -> np.dtype:
        """Return the numpy.dtype of this scene"""
        return np.uint8

    def get_crs_transformer(self):
        """Return the associated CRSTransformer."""
        return self.vector_source.crs_transformer

    def _get_chip(self, window):
        """Return the chip located in the window.

        Polygons falling within the window are rasterized using the class_id, and
        the background is filled with background_class_id. Also, any pixels in the
        window outside the extent are zero, which is the don't-care class for
        segmentation.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        log.debug(f'Rasterizing window: {window}')
        chip = geoms_to_raster(
            self.str_tree,
            window,
            background_class_id=self.background_class_id,
            extent=self.get_extent(),
            all_touched=self.all_touched)
        # Add third singleton dim since rasters must have >=1 channel.
        return np.expand_dims(chip, 2)

    def make_str_tree(self) -> STRtree:
        geojson = self.vector_source.get_geojson()
        self.validate_geojson(geojson)
        geoms = []
        for f in geojson['features']:
            geom = shape(f['geometry'])
            geom.class_id = f['properties']['class_id']
            geoms.append(geom)
        str_tree = STRtree(geoms)
        return str_tree

    def validate_geojson(self, geojson: dict) -> None:
        for f in geojson['features']:
            geom_type = f.get('geometry', {}).get('type', '')
            if 'Point' in geom_type or 'LineString' in geom_type:
                raise ValueError(
                    'LineStrings and Points are not supported '
                    'in RasterizedSource. Use BufferTransformer to '
                    'buffer them into Polygons.')
        for f in geojson['features']:
            if f.get('properties', {}).get('class_id') is None:
                raise ValueError('All GeoJSON features must have a class_id '
                                 'field in their properties.')
