import rastervision as rv
from rastervision.data import (LabelSource, LabelSourceConfig,
                               LabelSourceConfigBuilder,
                               ChipClassificationLabels)
from rastervision.protos.label_source_pb2 \
    import LabelSourceConfig as LabelSourceConfigMsg

NOOP_SOURCE = 'NOOP_SOURCE'


class NoopLabelSource(LabelSource):
    def get_labels(self, window=None):
        return ChipClassificationLabels()


class NoopLabelSourceConfig(LabelSourceConfig):
    def __init__(self):
        super().__init__(NOOP_SOURCE)

    def to_proto(self):
        msg = super().to_proto()
        msg.MergeFrom(LabelSourceConfigMsg(custom_config={}))
        return msg

    def create_source(self, task_config, extent, crs_transformer, tmp_dir):
        return NoopLabelSource()

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        return io_def or rv.core.CommandIODefinition()


class NoopLabelSourceConfigBuilder(LabelSourceConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(NoopLabelSourceConfig, {})

    def from_proto(self, msg):
        return self


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.LABEL_SOURCE, NOOP_SOURCE,
                                            NoopLabelSourceConfigBuilder)
