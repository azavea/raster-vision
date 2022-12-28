from enum import Enum
from typing import Optional

from rastervision.pipeline.config import register_config, Config, Field
from rastervision.core.rv_pipeline import RVPipelineConfig, PredictOptions
from rastervision.core.data.label_store import ObjectDetectionGeoJSONStoreConfig
from rastervision.core.evaluation import ObjectDetectionEvaluatorConfig


class ObjectDetectionWindowMethod(Enum):
    """Enum for window methods

    Attributes:
        chip: the default method
    """
    chip = 'chip'
    label = 'label'
    image = 'image'
    sliding = 'sliding'


@register_config('object_detection_chip_options')
class ObjectDetectionChipOptions(Config):
    neg_ratio: float = Field(
        1.0,
        description=
        ('The ratio of negative chips (those containing no bounding '
         'boxes) to positive chips. This can be useful if the statistics '
         'of the background is different in positive chips. For example, '
         'in car detection, the positive chips will always contain roads, '
         'but no examples of rooftops since cars tend to not be near rooftops.'
         ))
    ioa_thresh: float = Field(
        0.8,
        description=
        ('When a box is partially outside of a training chip, it is not clear if (a '
         'clipped version) of the box should be included in the chip. If the IOA '
         '(intersection over area) of the box with the chip is greater than ioa_thresh, '
         'it is included in the chip.'))
    window_method: ObjectDetectionWindowMethod = ObjectDetectionWindowMethod.chip
    label_buffer: Optional[int] = None


@register_config('object_detection_predict_options')
class ObjectDetectionPredictOptions(PredictOptions):
    merge_thresh: float = Field(
        0.5,
        description=
        ('If predicted boxes have an IOA (intersection over area) greater than '
         'merge_thresh, then they are merged into a single box during postprocessing. '
         'This is needed since the sliding window approach results in some false '
         'duplicates.'))
    score_thresh: float = Field(
        0.5,
        description=
        ('Predicted boxes are only output if their score is above score_thresh.'
         ))


@register_config('object_detection')
class ObjectDetectionConfig(RVPipelineConfig):
    """Configure an :class:`.ObjectDetection` pipeline."""

    chip_options: ObjectDetectionChipOptions = ObjectDetectionChipOptions()
    predict_options: ObjectDetectionPredictOptions = ObjectDetectionPredictOptions(
    )

    def build(self, tmp_dir):
        from rastervision.core.rv_pipeline.object_detection import ObjectDetection
        return ObjectDetection(self, tmp_dir)

    def get_default_label_store(self, scene):
        return ObjectDetectionGeoJSONStoreConfig()

    def get_default_evaluator(self):
        return ObjectDetectionEvaluatorConfig()
