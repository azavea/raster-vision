from copy import deepcopy

import rastervision as rv
from rastervision.data.raster_source.geotiff_source import GeoTiffSource
from rastervision.data.raster_source.raster_source_config \
    import (RasterSourceConfig, RasterSourceConfigBuilder)
from rastervision.protos.raster_source_pb2 \
    import RasterSourceConfig as RasterSourceConfigMsg
from rastervision.utils.files import download_if_needed


class GeoTiffSourceConfig(RasterSourceConfig):
    def __init__(self, uris, transformers=None, channel_order=None):
        super().__init__(
            source_type=rv.GEOTIFF_SOURCE,
            transformers=transformers,
            channel_order=channel_order)
        self.uris = uris

    def to_proto(self):
        msg = super().to_proto()
        msg.geotiff_files.CopyFrom(
            RasterSourceConfigMsg.GeoTiffFiles(uris=self.uris))
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
        return GeoTiffSource(self.uris, transformers, tmp_dir,
                             self.channel_order)

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        io_def = super().update_for_command(command_type, experiment_config,
                                            context, io_def)
        for uri in self.uris:
            io_def.add_input(uri)

        return io_def


class GeoTiffSourceConfigBuilder(RasterSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'uris': prev.uris,
                'transformers': prev.transformers,
                'channel_order': prev.channel_order
            }

        super().__init__(GeoTiffSourceConfig, config)

    def validate(self):
        super().validate()
        if self.config.get('uris') is None:
            raise rv.ConfigError(
                'You must specify uris for the GeoTiffSourceConfig. Use '
                '"with_uris".')

    def from_proto(self, msg):
        b = super().from_proto(msg)

        return b \
            .with_uris(msg.geotiff_files.uris)

    def with_uris(self, uris):
        """Set URIs for a GeoTIFFs containing as raster data."""
        b = deepcopy(self)
        b.config['uris'] = list(uris)
        return b

    def with_uri(self, uri):
        """Set URI for a GeoTIFF containing raster data."""
        b = deepcopy(self)
        b.config['uris'] = [uri]
        return b
