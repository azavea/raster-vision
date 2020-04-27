from enum import Enum

from rastervision2.pipeline.config import register_config, Config
from rastervision2.core.rv_pipeline import RVPipelineConfig
from rastervision2.core.data.label_store import ObjectDetectionGeoJSONStoreConfig
from rastervision2.core.evaluation import ObjectDetectionEvaluatorConfig


class ObjectDetectionWindowMethod(Enum):
    chip = 1


@register_config('object_detection_chip_options')
class ObjectDetectionChipOptions(Config):
    neg_ratio: float = 1.0
    ioa_thresh: float = 0.8
    window_method: ObjectDetectionWindowMethod = ObjectDetectionWindowMethod.chip
    label_buffer: float = 0.0


@register_config('object_detection_predict_options')
class ObjectDetectionPredictOptions(Config):
    merge_thresh: float = 0.5
    score_thresh: float = 0.5


@register_config('object_detection')
class ObjectDetectionConfig(RVPipelineConfig):
    chip_options: ObjectDetectionChipOptions = ObjectDetectionChipOptions()
    predict_options: ObjectDetectionPredictOptions = ObjectDetectionPredictOptions(
    )

    def build(self, tmp_dir):
        from rastervision2.core.rv_pipeline.object_detection import ObjectDetection
        return ObjectDetection(self, tmp_dir)

    def get_default_label_store(self, scene):
        return ObjectDetectionGeoJSONStoreConfig()

    def get_default_evaluator(self):
        return ObjectDetectionEvaluatorConfig()
