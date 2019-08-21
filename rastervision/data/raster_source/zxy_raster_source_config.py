from copy import deepcopy

import rastervision as rv
from rastervision.data.raster_source.zxy_raster_source import ZXYRasterSource
from rastervision.data.raster_source.raster_source_config \
    import (RasterSourceConfig, RasterSourceConfigBuilder)
from rastervision.protos.raster_source_pb2 \
    import RasterSourceConfig as RasterSourceConfigMsg


class ZXYRasterSourceConfig(RasterSourceConfig):
    def __init__(self,
                 tile_schema,
                 zoom,
                 bounds,
                 x_shift_meters=0.0,
                 y_shift_meters=0.0,
                 transformers=None,
                 channel_order=None):
        """Construct the configuration for a ZXYRasterSource.

        Args:
            tile_schema: (str) the URI schema for zxy tiles (ie. a slippy map tile server)
                of the form /tileserver-uri/{z}/{x}/{y}.png. If {-y} is used, the tiles
                are assumed to be indexed using TMS coordinates, where the y axis starts
                at the southernmost point. The URI can be for http, S3, or the local
                file system.
            zoom: (int) the zoom level to use when retrieving tiles
            bounds: (list) a list of length 4 containing min_lat, min_lng,
                max_lat, max_lng
            x_shift_meters: (float) A number of meters to shift along the
                x-axis. A ositive shift moves the "camera" to the right.
            y_shift_meters: A number of meters to shift along the y-axis. A
                positive shift moves the "camera" down.
            transformers: list of RasterTransformers to apply
            channel_order: list of indices of channels to extract from raw
                imagery
        """
        super().__init__(
            source_type=rv.ZXY_RASTER_SOURCE,
            transformers=transformers,
            channel_order=channel_order)
        self.tile_schema = tile_schema
        self.zoom = zoom
        self.bounds = bounds
        self.x_shift_meters = x_shift_meters
        self.y_shift_meters = y_shift_meters

    def to_proto(self):
        msg = super().to_proto()
        msg.zxy_raster_source.CopyFrom(
            RasterSourceConfigMsg.ZXYRasterSource(
                tile_schema=self.tile_schema,
                zoom=self.zoom,
                bounds=self.bounds,
                x_shift_meters=self.x_shift_meters,
                y_shift_meters=self.y_shift_meters))
        return msg

    # These three methods are not implemented because the
    # ZXYRasterSource is incompatible with the predict command
    # as it is currently implemented.
    def save_bundle_files(self, bundle_dir):
        raise NotImplementedError()

    def for_prediction(self, image_uri):
        raise NotImplementedError()

    def create_local(self, tmp_dir):
        raise NotImplementedError()

    def create_source(self,
                      tmp_dir,
                      crs_transformer=None,
                      extent=None,
                      class_map=None):
        transformers = self.create_transformers()
        x_shift_meters = self.x_shift_meters
        y_shift_meters = self.y_shift_meters
        return ZXYRasterSource(
            tile_schema=self.tile_schema,
            zoom=self.zoom,
            bounds=self.bounds,
            raster_transformers=transformers,
            temp_dir=tmp_dir,
            channel_order=self.channel_order,
            x_shift_meters=x_shift_meters,
            y_shift_meters=y_shift_meters)

    def report_io(self, command_type, io_def):
        super().report_io(command_type, io_def)


class ZXYRasterSourceConfigBuilder(RasterSourceConfigBuilder):
    """This RasterSource can read from Z/X/Y tile servers."""

    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'tile_schema': prev.tile_schema,
                'zoom': prev.zoom,
                'bounds': prev.bounds,
                'transformers': prev.transformers,
                'channel_order': prev.channel_order,
                'x_shift_meters': prev.x_shift_meters,
                'y_shift_meters': prev.y_shift_meters,
            }

        super().__init__(ZXYRasterSourceConfig, config)

    def validate(self):
        super().validate()
        tile_schema = self.config.get('tile_schema')
        zoom = self.config.get('zoom')
        bounds = self.config.get('bounds')
        if tile_schema is None:
            raise rv.ConfigError(
                'Must set tile_schema using with_tile_schema().')
        if zoom is None:
            raise rv.ConfigError('Must set zoom level using with_zoom().')
        if bounds is None:
            raise rv.ConfigError('Must set bounds using with_bounds().')

        if '{z}/{x}/{y}' not in tile_schema and '{z}/{x}/{-y}' not in tile_schema:
            raise rv.ConfigError(
                'tile_schema must contain {z}/{x}/{y} or {z}/{x}/{-y}')

        if len(bounds) == 4:
            min_lat, min_lng, max_lat, max_lng = bounds
            if min_lat >= max_lat or min_lng >= max_lng:
                raise rv.ConfigError(
                    'bounds should be of form [min_lat, min_lng, max_lat, max_lng]'
                )
        else:
            raise rv.ConfigError(
                'bounds should be of form [min_lat, min_lng, max_lat, max_lng]'
            )

    def from_proto(self, msg):
        b = super().from_proto(msg)
        source = msg.zxy_raster_source
        return b \
            .with_tile_schema(source.tile_schema) \
            .with_zoom(source.zoom) \
            .with_bounds(source.bounds) \
            .with_shifts(source.x_shift_meters, source.y_shift_meters)

    def with_tile_schema(self, tile_schema):
        """Set the tile schema

        Args:
            tile_schema: (str) the URI schema for zxy tiles (ie. a slippy map
                tile server) of the form /tileserver-uri/{z}/{x}/{y}.png.
                If {-y} is used, the tiles are assumed to be indexed using
                TMS coordinates, where the y axis starts at the southernmost
                point. The URI can be for http, S3, or the local file system.
        """
        b = deepcopy(self)
        b.config['tile_schema'] = tile_schema
        return b

    def with_zoom(self, zoom):
        """Set the zoom level to use when retrieving tiles.

        Args:
            zoom: (int) the zoom level to use when retrieving tiles
        """
        b = deepcopy(self)
        b.config['zoom'] = zoom
        return b

    def with_bounds(self, bounds):
        """Set the bounds to use for retrieving imagery from the tiles.

        Args:
            bounds: (list) a list of length 4 containing min_lat, min_lng,
                max_lat, max_lng
        """
        b = deepcopy(self)
        b.config['bounds'] = bounds
        return b

    def with_shifts(self, x, y):
        """Set the x- and y-shifts in meters.

        Args:
            x: A number of meters to shift along the x-axis.  A
                positive shift moves the "camera" to the right.

            y: A number of meters to shift along the y-axis.  A
                positive shift moves the "camera" down.
        """
        b = deepcopy(self)
        b.config['x_shift_meters'] = x
        b.config['y_shift_meters'] = y
        return b
