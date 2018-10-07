from copy import deepcopy

import rastervision as rv
from rastervision.core.class_map import ClassMap
from rastervision.data.label_source import (LabelSourceConfig,
                                            LabelSourceConfigBuilder,
                                            SemanticSegmentationRasterSource)
from rastervision.protos.label_source_pb2 import LabelSourceConfig as LabelSourceConfigMsg
from rastervision.data.raster_source import RasterSourceConfig, GeoJSONSourceConfig


class SemanticSegmentationRasterSourceConfig(LabelSourceConfig):
    def __init__(self, source, rgb_class_map=None):
        super().__init__(source_type=rv.SEMANTIC_SEGMENTATION_RASTER)
        self.source = source
        self.rgb_class_map = rgb_class_map

    def to_proto(self):
        msg = super().to_proto()

        rgb_class_items = None
        if self.rgb_class_map is not None:
            rgb_class_items = self.rgb_class_map.to_proto()
        opts = LabelSourceConfigMsg.SemanticSegmentationRasterSource(
            source=self.source.to_proto(), rgb_class_items=rgb_class_items)
        msg.semantic_segmentation_raster_source.CopyFrom(opts)
        return msg

    def create_source(self, task_config, extent, crs_transformer, tmp_dir):
        return SemanticSegmentationRasterSource(
            self.source.create_source(tmp_dir, extent, crs_transformer),
            self.rgb_class_map)

    def update_for_command(self, command_type, experiment_config, context=[]):
        if context is None:
            context = []
        context = context + [self]
        io_def = rv.core.CommandIODefinition()

        b = self.to_builder()

        (new_raster_source, sub_io_def) = self.source.update_for_command(
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
                'rgb_class_map': prev.rgb_class_map
            }

        super().__init__(SemanticSegmentationRasterSourceConfig, config)

    def from_proto(self, msg):
        b = SemanticSegmentationRasterSourceConfigBuilder()

        raster_source_config = rv.RasterSourceConfig.from_proto(
            msg.semantic_segmentation_raster_source.source)

        b = b.with_raster_source(raster_source_config)
        rgb_class_items = msg.semantic_segmentation_raster_source.rgb_class_items
        if rgb_class_items:
            b = b.with_rgb_class_map(
                ClassMap.construct_from(list(rgb_class_items)))

        return b

    def with_raster_source(self, source, channel_order=None):
        """Set raster_source.

        Args:
            source: (RasterSourceConfig) A RasterSource assumed to have RGB values that
                are mapped to class_ids using the rgb_class_map.

        Returns:
            SemanticSegmentationRasterSourceConfigBuilder
        """
        b = deepcopy(self)
        if isinstance(source, RasterSourceConfig):
            b.config['source'] = source
        elif isinstance(source, str):
            provider = rv._registry.get_raster_source_default_provider(source)
            source = provider.construct(source, channel_order=channel_order)
            b.config['source'] = source
        else:
            raise rv.ConfigError(
                'source must be either string or RasterSourceConfig, '
                ' not {}'.format(str(type(source))))

        return b

    def with_rgb_class_map(self, rgb_class_map):
        """Set rgb_class_map.

        Args:
            rgb_class_map: (something accepted by ClassMap.construct_from) a class
                map with color values used to map RGB values to class ids

        Returns:
            SemanticSegmentationRasterSourceConfigBuilder
        """
        b = deepcopy(self)
        b.config['rgb_class_map'] = ClassMap.construct_from(rgb_class_map)
        return b

    def validate(self):
        source = self.config.get('source')
        rgb_class_map = self.config.get('rgb_class_map')

        if source is None:
            raise rv.ConfigError(
                'You must set the source for SemanticSegmentationRasterSourceConfig'
                ' Use "with_raster_source".')

        if type(source) != GeoJSONSourceConfig and rgb_class_map is None:
            raise rv.ConfigError(
                'You must set the rgb_class_map for '
                'SemanticSegmentationRasterSourceConfig. Use "with_rgb_class_map".'
            )
