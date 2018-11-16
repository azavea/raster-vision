from copy import deepcopy

import rastervision as rv
from rastervision.data.raster_source.image_source import ImageSource
from rastervision.data.raster_source.raster_source_config \
    import (RasterSourceConfig, RasterSourceConfigBuilder)
from rastervision.protos.raster_source_pb2 \
    import RasterSourceConfig as RasterSourceConfigMsg
from rastervision.utils.files import download_if_needed


class ImageSourceConfig(RasterSourceConfig):
    def __init__(self, uri, transformers=None, channel_order=None):
        super().__init__(
            source_type=rv.IMAGE_SOURCE,
            transformers=transformers,
            channel_order=channel_order)
        self.uri = uri

    def to_proto(self):
        msg = super().to_proto()
        msg.MergeFrom(
            RasterSourceConfigMsg(
                image_file=RasterSourceConfigMsg.ImageFile(uri=self.uri)))
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

    def create_source(self,
                      tmp_dir,
                      crs_transformer=None,
                      extent=None,
                      class_map=None):
        transformers = self.create_transformers()
        return ImageSource(self.uri, transformers, tmp_dir, self.channel_order)

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        io_def = super().update_for_command(command_type, experiment_config,
                                            io_def)
        io_def.add_input(self.uri)

        return io_def


class ImageSourceConfigBuilder(RasterSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'uri': prev.uri,
                'transformers': prev.transformers,
                'channel_order': prev.channel_order
            }

        super().__init__(ImageSourceConfig, config)

    def validate(self):
        super().validate()
        if self.config.get('uri') is None:
            raise rv.ConfigError(
                'You must specify a uri for the ImageSourceConfig. Use "with_uri"'
            )

    def from_proto(self, msg):
        b = super().from_proto(msg)

        return b \
            .with_uri(msg.image_file.uri)

    def with_uri(self, uri):
        """Set URI for an image.

        Args:
            uri: A URI pointing to some (non-georeferenced) raster
                file (TIFs, PNGs, and JPEGs are supported, and
                possibly others).

        """
        b = deepcopy(self)
        b.config['uri'] = uri
        return b
