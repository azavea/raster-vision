from copy import deepcopy

import rastervision as rv
from rastervision.core.class_map import ClassMap
from rastervision.data.label_source import (LabelSourceConfig,
                                            LabelSourceConfigBuilder,
                                            SemanticSegmentationRasterSource)
from rastervision.protos.label_source_pb2 import LabelSourceConfig as LabelSourceConfigMsg
from rastervision.data.raster_source import RasterSourceConfig


class SemanticSegmentationRasterSourceConfig(LabelSourceConfig):
    def __init__(self, source, source_class_map):
        super().__init__(source_type=rv.SEMANTIC_SEGMENTATION_RASTER)
        self.source = source
        self.source_class_map = source_class_map

    def to_proto(self):
        msg = super().to_proto()
        opts = LabelSourceConfigMsg.SemanticSegmentationRasterSource(
            source=self.source.to_proto(),
            source_class_items=self.source_class_map.to_proto())
        msg.semantic_segmentation_raster_source.CopyFrom(opts)
        return msg

    def create_source(self, task_config, extent, crs_transformer, tmp_dir):
        return SemanticSegmentationRasterSource(
            self.source.create_source(tmp_dir), self.source_class_map)

    def preprocess_command(self, command_type, experiment_config, context=[]):
        if context is None:
            context = []
        context = context + [self]
        io_def = rv.core.CommandIODefinition()

        b = self.to_builder()

        (new_raster_source, sub_io_def) = self.source.preprocess_command(
            command_type, experiment_config, context)
        io_def.merge(sub_io_def)
        b = b.with_raster_source(new_raster_source)

        return (b.build(), io_def)


class SemanticSegmentationRasterSourceConfigBuilder(LabelSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'source': prev.source,
                'source_class_map': prev.source_class_map
            }

        super().__init__(SemanticSegmentationRasterSourceConfig, config)

    def from_proto(self, msg):
        b = SemanticSegmentationRasterSourceConfigBuilder()

        raster_source_config = rv.RasterSourceConfig.from_proto(
            msg.semantic_segmentation_raster_source.source)

        return b \
            .with_raster_source(raster_source_config) \
            .with_source_class_map(
                ClassMap.construct_from(
                    list(msg.semantic_segmentation_raster_source.source_class_items)))

    def with_raster_source(self, source, channel_order=None):
        """Set raster_source.

        Args:
            source: (RasterSourceConfig) A RasterSource assumed to have RGB values that
                are mapped to class_ids using the source_class_map.

        Returns:
            SemanticSegmentationRasterSourceConfigBuilder
        """
        b = deepcopy(self)
        if isinstance(source, RasterSourceConfig):
            b.config['source'] = source
        elif isinstance(source, str):
            provider = rv._registry.get_default_raster_source_provider(source)
            b.config['source'] = provider.construct(
                source, channel_order=channel_order)
        else:
            raise rv.ConfigError(
                'source must be either string or RasterSourceConfig, '
                ' not {}'.format(str(type(source))))

        return b

    def with_source_class_map(self, source_class_map):
        """Set source_class_map.

        Args:
            source_class_map: (something accepted by ClassMap.construct_from) a class
                map with color values used to map RGB values to class ids

        Returns:
            SemanticSegmentationRasterSourceConfigBuilder
        """
        b = deepcopy(self)
        b.config['source_class_map'] = ClassMap.construct_from(
            source_class_map)
        return b

    def validate(self):
        if self.config.get('source') is None:
            raise rv.ConfigError(
                'You must set the source for SemanticSegmentationRasterSourceConfig'
                ' Use "with_raster_source".')

        if self.config.get('source_class_map') is None:
            raise rv.ConfigError(
                'You must set the source_class_map for '
                'SemanticSegmentationRasterSourceConfig. Use "with_source_class_map".'
            )
