from copy import deepcopy

import rastervision as rv
from rastervision.data.raster_source.rasterio_source import RasterioSource
from rastervision.data.raster_source.raster_source_config \
    import (RasterSourceConfig, RasterSourceConfigBuilder)
from rastervision.protos.raster_source_pb2 \
    import RasterSourceConfig as RasterSourceConfigMsg
from rastervision.utils.files import download_if_needed


class RasterioSourceConfig(RasterSourceConfig):
    def __init__(self,
                 uris,
                 x_shift_meters=0.0,
                 y_shift_meters=0.0,
                 transformers=None,
                 channel_order=None):
        super().__init__(
            source_type=rv.RASTERIO_SOURCE,
            transformers=transformers,
            channel_order=channel_order)
        self.uris = uris
        self.x_shift_meters = x_shift_meters
        self.y_shift_meters = y_shift_meters

    def to_proto(self):
        msg = super().to_proto()
        msg.rasterio_source.CopyFrom(
            RasterSourceConfigMsg.RasterioSource(
                uris=self.uris,
                x_shift_meters=self.x_shift_meters,
                y_shift_meters=self.y_shift_meters))
        return msg

    def save_bundle_files(self, bundle_dir):
        (conf, files) = super().save_bundle_files(bundle_dir)

        # Replace the URI with a template value.
        new_config = conf.to_builder() \
                         .with_uri('BUNDLE') \
                         .build()
        return (new_config, files)

    def for_prediction(self, image_uri):
        return self.to_builder() \
                   .with_uri(image_uri) \
                   .build()

    def create_local(self, tmp_dir):
        new_uris = [download_if_needed(uri, tmp_dir) for uri in self.uris]
        return self.to_builder() \
                   .with_uris(new_uris) \
                   .build()

    def create_source(self,
                      tmp_dir,
                      crs_transformer=None,
                      extent=None,
                      class_map=None):
        transformers = self.create_transformers()
        x_shift_meters = self.x_shift_meters
        y_shift_meters = self.y_shift_meters
        return RasterioSource(
            uris=self.uris,
            raster_transformers=transformers,
            temp_dir=tmp_dir,
            channel_order=self.channel_order,
            x_shift_meters=x_shift_meters,
            y_shift_meters=y_shift_meters)

    def report_io(self, command_type, io_def):
        super().report_io(command_type, io_def)
        io_def.add_inputs(self.uris)


class RasterioSourceConfigBuilder(RasterSourceConfigBuilder):
    """This RasterSource can read any file that can be opened by Rasterio/GDAL.

    This includes georeferenced formats such as GeoTIFF and non-georeferenced formats
    such as JPG. See https://www.gdal.org/formats_list.html for more details.
    """

    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'uris': prev.uris,
                'transformers': prev.transformers,
                'channel_order': prev.channel_order,
                'x_shift_meters': prev.x_shift_meters,
                'y_shift_meters': prev.y_shift_meters,
            }

        super().__init__(RasterioSourceConfig, config)

    def validate(self):
        super().validate()
        if self.config.get('uris') is None:
            raise rv.ConfigError(
                'You must specify uris for the RasterioSourceConfig. Use '
                '"with_uris".')
        if not isinstance(self.config.get('uris'), list):
            raise rv.ConfigError(
                'uris set with "with_uris" must be a list, got {}'.format(
                    type(self.config.get('uris'))))
        for uri in self.config.get('uris'):
            if not isinstance(uri, str):
                raise rv.ConfigError('uri must be a string, got {}'.format(
                    type(uri)))

    def from_proto(self, msg):
        b = super().from_proto(msg)

        # Need to do this for backward compatibility.
        if msg.HasField('geotiff_files') or msg.HasField('rasterio_source'):
            if msg.HasField('geotiff_files'):
                source = msg.geotiff_files
            else:
                source = msg.rasterio_source

            return b \
                .with_uris(source.uris) \
                .with_shifts(source.x_shift_meters, source.y_shift_meters)
        elif msg.HasField('image_file'):
            source = msg.image_file
            return b.with_uri(source.uri)
        else:
            raise rv.ConfigError(
                'RasterioSourceConfig protobuf message should contain geotiff_files, '
                'rasterio_source, or image_file')

    def with_uris(self, uris):
        """Set URIs for raster files that can be read by Rasterio."""
        b = deepcopy(self)
        b.config['uris'] = list(uris)
        return b

    def with_uri(self, uri):
        """Set URI for raster files that can be read by Rasterio."""
        b = deepcopy(self)
        b.config['uris'] = [uri]
        return b

    def with_shifts(self, x, y):
        """Set the x- and y-shifts in meters.

        This will only have an effect on georeferenced imagery.

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
