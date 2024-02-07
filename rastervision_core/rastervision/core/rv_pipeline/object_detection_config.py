from typing import Optional

from rastervision.pipeline.config import Field, register_config, validator
from rastervision.core.rv_pipeline import (
    ChipOptions, RVPipelineConfig, PredictOptions, WindowSamplingConfig)
from rastervision.core.data.label_store import ObjectDetectionGeoJSONStoreConfig
from rastervision.core.evaluation import ObjectDetectionEvaluatorConfig


@register_config('object_detection_window_sampling')
class ObjectDetectionWindowSamplingConfig(WindowSamplingConfig):
    ioa_thresh: float = Field(
        0.8,
        description='When a box is partially outside of a training chip, it '
        'is not clear if (a clipped version) of the box should be included in '
        'the chip. If the IOA (intersection over area) of the box with the '
        'chip is greater than ioa_thresh, it is included in the chip. '
        'Defaults to 0.8.')
    clip: bool = Field(
        False,
        description='Clip bounding boxes to window limits when retrieving '
        'labels for a window.')
    neg_ratio: Optional[float] = Field(
        None,
        description='The ratio of negative chips (those containing no '
        'bounding boxes) to positive chips. This can be useful if the '
        'statistics of the background is different in positive chips. For '
        'example, in car detection, the positive chips will always contain '
        'roads, but no examples of rooftops since cars tend to not be near '
        'rooftops. Defaults to None.')
    neg_ioa_thresh: float = Field(
        0.2,
        description='A window will be considered negative if its max IoA with '
        'any bounding box is less than this threshold. Defaults to 0.2.')


@register_config('object_detection_chip_options')
class ObjectDetectionChipOptions(ChipOptions):
    sampling: ObjectDetectionWindowSamplingConfig = Field(
        ..., description='Window sampling config.')


@register_config('object_detection_predict_options')
class ObjectDetectionPredictOptions(PredictOptions):
    stride: Optional[int] = Field(
        None,
        description='Stride of the sliding window for generating chips. '
        'Defaults to half of ``chip_sz``.')
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

    @validator('stride', always=True)
    def validate_stride(cls, v: Optional[int], values: dict) -> dict:
        if v is None:
            chip_sz: int = values['chip_sz']
            return chip_sz // 2
        return v


@register_config('object_detection')
class ObjectDetectionConfig(RVPipelineConfig):
    """Configure an :class:`.ObjectDetection` pipeline."""

    chip_options: Optional[ObjectDetectionChipOptions]
    predict_options: Optional[ObjectDetectionPredictOptions]

    def build(self, tmp_dir):
        from rastervision.core.rv_pipeline.object_detection import ObjectDetection
        return ObjectDetection(self, tmp_dir)

    def get_default_label_store(self, scene):
        return ObjectDetectionGeoJSONStoreConfig()

    def get_default_evaluator(self):
        return ObjectDetectionEvaluatorConfig()
