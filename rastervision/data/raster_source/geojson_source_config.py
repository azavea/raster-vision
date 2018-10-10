from copy import deepcopy

import rastervision as rv
from rastervision.data.raster_source.geojson_source import GeoJSONSource
from rastervision.data.raster_source.raster_source_config \
    import (RasterSourceConfig, RasterSourceConfigBuilder)
from rastervision.protos.raster_source_pb2 \
    import RasterSourceConfig as RasterSourceConfigMsg
from rastervision.utils.files import download_if_needed


class GeoJSONSourceConfig(RasterSourceConfig):
    class RasterizerOptions(object):
        def __init__(self, background_class_id, line_buffer=15):
            """Constructor.

            Args:
                background_class_id: The class_id to use for background pixels that don't
                    overlap with any shapes in the GeoJSON file.
                line_buffer: Number of pixels to add to each side of line when rasterized.
            """
            self.background_class_id = background_class_id
            self.line_buffer = line_buffer

        def to_proto(self):
            return RasterSourceConfigMsg.GeoJSONFile.RasterizerOptions(
                background_class_id=self.background_class_id,
                line_buffer=self.line_buffer)

    def __init__(self,
                 uri,
                 rasterizer_options,
                 transformers=None,
                 channel_order=None):
        super().__init__(
            source_type=rv.GEOJSON_SOURCE,
            transformers=transformers,
            channel_order=channel_order)
        self.uri = uri
        self.rasterizer_options = rasterizer_options

    def to_proto(self):
        msg = super().to_proto()
        msg.MergeFrom(
            RasterSourceConfigMsg(
                geojson_file=RasterSourceConfigMsg.GeoJSONFile(
                    uri=self.uri,
                    rasterizer_options=self.rasterizer_options.to_proto())))
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
        new_uri = download_if_needed(self.uri, tmp_dir)
        return self.to_builder() \
                   .with_uri(new_uri) \
                   .build()

    def create_source(self, tmp_dir, extent, crs_transformer):
        return GeoJSONSource(self.uri, self.rasterizer_options, extent,
                             crs_transformer)

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        (conf, io_def) = super().update_for_command(command_type,
                                                    experiment_config)
        io_def.add_input(self.uri)

        return (conf, io_def)


class GeoJSONSourceConfigBuilder(RasterSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'uri': prev.uri,
                'rasterizer_options': prev.rasterizer_options
            }

        super().__init__(GeoJSONSourceConfig, config)

    def validate(self):
        super().validate()
        if self.config.get('uri') is None:
            raise rv.ConfigError(
                'You must specify a uri for the GeoJSONSourceConfig. Use "with_uri"'
            )

        if self.config.get('rasterizer_options') is None:
            raise rv.ConfigError(
                'You must configure the rasterizer for the GeoJSONSourceConfig. '
                'Use "with_rasterizer_options"')

    def from_proto(self, msg):
        b = super().from_proto(msg)
        return b \
            .with_uri(msg.geojson_file.uri) \
            .with_rasterizer_options(
                msg.geojson_file.rasterizer_options.background_class_id,
                msg.geojson_file.rasterizer_options.line_buffer)

    def with_uri(self, uri):
        """Set URI for a GeoJSON file used to read labels."""
        b = deepcopy(self)
        b.config['uri'] = uri
        return b

    def with_rasterizer_options(self, background_class_id, line_buffer=15):
        """Specify options for converting GeoJSON to raster.

        Args:
            background_class_id: The class_id to use for background pixels that don't
                overlap with any shapes in the GeoJSON file.
            line_buffer: Number of pixels to add to each side of line when rasterized.
        """
        b = deepcopy(self)
        b.config['rasterizer_options'] = GeoJSONSourceConfig.RasterizerOptions(
            background_class_id, line_buffer=line_buffer)
        return b
