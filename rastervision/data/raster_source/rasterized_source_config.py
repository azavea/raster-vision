from copy import deepcopy

import rastervision as rv
from rastervision.data.raster_source.rasterized_source import (
    RasterizedSource)
from rastervision.data.raster_source.raster_source_config \
    import (RasterSourceConfig, RasterSourceConfigBuilder)
from rastervision.protos.raster_source_pb2 \
    import RasterSourceConfig as RasterSourceConfigMsg
from rastervision.data.vector_source import VectorSourceConfig
from rastervision.utils.files import download_if_needed


class RasterizedSourceConfig(RasterSourceConfig):
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
            return RasterSourceConfigMsg.RasterizedSource.RasterizerOptions(
                background_class_id=self.background_class_id,
                line_buffer=self.line_buffer)

    def __init__(self,
                 vector_source,
                 rasterizer_options,
                 transformers=None,
                 channel_order=None):
        super().__init__(
            source_type=rv.RASTERIZED_SOURCE,
            transformers=transformers,
            channel_order=channel_order)
        self.vector_source = vector_source
        self.rasterizer_options = rasterizer_options

    def to_proto(self):
        msg = super().to_proto()
        msg.MergeFrom(
            RasterSourceConfigMsg(
                rasterized_source=RasterSourceConfigMsg.RasterizedSource(
                    vector_source=self.vector_source.to_proto(),
                    rasterizer_options=self.rasterizer_options.to_proto())))
        return msg

    def save_bundle_files(self, bundle_dir):
        (conf, files) = super().save_bundle_files(bundle_dir)
        # Replace the URI with a template value.
        new_config = conf.to_builder() \
                         .with_uri('BUNDLE') \
                         .build()
        return (new_config, files)

    def for_prediction(self, uri):
        return self.to_builder() \
                   .with_uri(uri) \
                   .build()

    def create_local(self, tmp_dir):
        new_uri = download_if_needed(self.uri, tmp_dir)
        return self.to_builder() \
                   .with_uri(new_uri) \
                   .build()

    def create_source(self, tmp_dir, crs_transformer, extent, class_map=None):
        vector_source = self.vector_source.create_source(
            crs_transformer=crs_transformer,
            extent=extent,
            class_map=class_map)
        return RasterizedSource(vector_source, self.rasterizer_options, extent,
                                crs_transformer)

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        io_def = super().update_for_command(command_type, experiment_config,
                                            context, io_def)
        self.vector_source.update_for_command(command_type, experiment_config,
                                              context, io_def)
        return io_def


class RasterizedSourceConfigBuilder(RasterSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'vector_source': prev.vector_source,
                'rasterizer_options': prev.rasterizer_options
            }

        super().__init__(RasterizedSourceConfig, config)

    def validate(self):
        super().validate()
        if self.config.get('vector_source') is None:
            raise rv.ConfigError(
                'You must specify a vector_source for the RasterizedSourceConfig. '
                'Use "with_vector_source"')

        if self.config.get('rasterizer_options') is None:
            raise rv.ConfigError(
                'You must configure the rasterizer for the RasterizedSourceConfig. '
                'Use "with_rasterizer_options"')

    def from_proto(self, msg):
        b = super().from_proto(msg)
        vector_source = VectorSourceConfig.from_proto(
            msg.rasterized_source.vector_source)
        return b \
            .with_vector_source(vector_source) \
            .with_rasterizer_options(
                msg.rasterized_source.rasterizer_options.background_class_id,
                msg.rasterized_source.rasterizer_options.line_buffer)

    def with_vector_source(self, vector_source):
        """Set the vector_source.

        Args:
            vector_source (str or VectorSource) if a string, assume it is
                a URI and use the default provider to construct a VectorSource.
        """
        if isinstance(vector_source, str):
            return self.with_uri(vector_source)

        b = deepcopy(self)
        if isinstance(vector_source, VectorSourceConfig):
            b.config['vector_source'] = vector_source
        else:
            raise rv.ConfigError(
                'vector_source must be of type str or VectorSource')

        return b

    def with_uri(self, uri):
        b = deepcopy(self)
        provider = rv._registry.get_vector_source_default_provider(uri)
        b.config['vector_source'] = provider.construct(uri)
        return b

    def with_rasterizer_options(self, background_class_id, line_buffer=15):
        """Specify options for converting GeoJSON to raster.

        Args:
            background_class_id: The class_id to use for background pixels that don't
                overlap with any shapes in the GeoJSON file.
            line_buffer: Number of pixels to add to each side of line when rasterized.
        """
        b = deepcopy(self)
        b.config[
            'rasterizer_options'] = RasterizedSourceConfig.RasterizerOptions(
                background_class_id, line_buffer=line_buffer)
        return b
