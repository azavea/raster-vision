# flake8: noqa
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rastervision.pipeline import Registry


def register_plugin(registry: 'Registry'):
    registry.set_plugin_version('rastervision.pytorch_learner', 7)
    registry.register_renamed_type_hints('geo_data_window', 'window_sampling')


import rastervision.pipeline
from rastervision.pytorch_learner.learner_config import *
from rastervision.pytorch_learner.learner import *
from rastervision.pytorch_learner.classification_learner_config import *
from rastervision.pytorch_learner.classification_learner import *
from rastervision.pytorch_learner.regression_learner_config import *
from rastervision.pytorch_learner.regression_learner import *
from rastervision.pytorch_learner.semantic_segmentation_learner_config import *
from rastervision.pytorch_learner.semantic_segmentation_learner import *
from rastervision.pytorch_learner.object_detection_learner_config import *
from rastervision.pytorch_learner.object_detection_learner import *
from rastervision.pytorch_learner.dataset import *

__all__ = [
    # Learner
    Learner.__name__,
    SemanticSegmentationLearner.__name__,
    ClassificationLearner.__name__,
    ObjectDetectionLearner.__name__,
    RegressionLearner.__name__,
    # LearnerConfig
    LearnerConfig.__name__,
    SemanticSegmentationLearnerConfig.__name__,
    ClassificationLearnerConfig.__name__,
    ObjectDetectionLearnerConfig.__name__,
    RegressionLearnerConfig.__name__,
    # DataConfig
    DataConfig.__name__,
    GeoDataConfig.__name__,
    PlotOptions.__name__,
    ImageDataConfig.__name__,
    SemanticSegmentationDataConfig.__name__,
    SemanticSegmentationGeoDataConfig.__name__,
    SemanticSegmentationImageDataConfig.__name__,
    ClassificationDataConfig.__name__,
    ClassificationGeoDataConfig.__name__,
    ClassificationImageDataConfig.__name__,
    ObjectDetectionDataConfig.__name__,
    ObjectDetectionGeoDataConfig.__name__,
    ObjectDetectionImageDataConfig.__name__,
    RegressionDataConfig.__name__,
    RegressionGeoDataConfig.__name__,
    RegressionImageDataConfig.__name__,
    PlotOptions.__name__,
    # DataFormat
    SemanticSegmentationDataFormat.__name__,
    ClassificationDataFormat.__name__,
    ObjectDetectionDataFormat.__name__,
    RegressionDataFormat.__name__,
    # ModelConfig
    ModelConfig.__name__,
    ExternalModuleConfig.__name__,
    Backbone.__name__,
    SemanticSegmentationModelConfig.__name__,
    ClassificationModelConfig.__name__,
    ObjectDetectionModelConfig.__name__,
    RegressionModelConfig.__name__,
    # SolverConfig
    SolverConfig.__name__,
    # Dataset
    AlbumentationsDataset.__name__,
    GeoDataset.__name__,
    ImageDataset.__name__,
    SlidingWindowGeoDataset.__name__,
    RandomWindowGeoDataset.__name__,
    SemanticSegmentationSlidingWindowGeoDataset.__name__,
    SemanticSegmentationRandomWindowGeoDataset.__name__,
    SemanticSegmentationImageDataset.__name__,
    SemanticSegmentationDataReader.__name__,
    ClassificationSlidingWindowGeoDataset.__name__,
    ClassificationRandomWindowGeoDataset.__name__,
    ClassificationImageDataset.__name__,
    ObjectDetectionSlidingWindowGeoDataset.__name__,
    ObjectDetectionRandomWindowGeoDataset.__name__,
    ObjectDetectionImageDataset.__name__,
    CocoDataset.__name__,
    RegressionSlidingWindowGeoDataset.__name__,
    RegressionRandomWindowGeoDataset.__name__,
    RegressionImageDataset.__name__,
    RegressionDataReader.__name__,
    TransformType.__name__,
    DatasetError.__name__,
    ImageDatasetError.__name__,
    GeoDatasetError.__name__,
    # Visualizer
    Visualizer.__name__,
    SemanticSegmentationVisualizer.__name__,
    ClassificationVisualizer.__name__,
    ObjectDetectionVisualizer.__name__,
    RegressionVisualizer.__name__,
]
