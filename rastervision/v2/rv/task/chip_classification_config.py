from rastervision.v2.rv.task import TaskConfig
from rastervision.v2.core.config import register_config

@register_config('chip_classification')
class ChipClassificationConfig(TaskConfig):
    def get_pipeline(self):
        from rastervision.v2.rv.task.chip_classification import ChipClassification
        return ChipClassification
