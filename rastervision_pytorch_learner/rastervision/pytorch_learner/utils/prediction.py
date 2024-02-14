from typing import TYPE_CHECKING, Dict, Iterator
import logging

from rastervision.core.data import (ChipClassificationLabels,
                                    ObjectDetectionLabels,
                                    SemanticSegmentationLabels)

if TYPE_CHECKING:
    import numpy as np
    from rastervision.core.data import Scene, SemanticSegmentationLabelStore
    from rastervision.core.rv_pipeline import (
        PredictOptions, ObjectDetectionPredictOptions,
        SemanticSegmentationPredictOptions)
    from rastervision.pytorch_learner import (ClassificationLearner,
                                              ObjectDetectionLearner,
                                              SemanticSegmentationLearner)

log = logging.getLogger(__name__)


def predict_scene_cc(
        learner: 'ClassificationLearner', scene: 'Scene',
        predict_options: 'PredictOptions') -> 'ChipClassificationLabels':
    """Generate chip classification predictions for a :class:`.Scene`."""
    from rastervision.pytorch_learner.dataset import (
        ClassificationSlidingWindowGeoDataset)

    chip_sz = predict_options.chip_sz
    stride = predict_options.stride
    batch_sz = predict_options.batch_sz

    base_tf, _ = learner.cfg.data.get_data_transforms()
    ds = ClassificationSlidingWindowGeoDataset(
        scene, size=chip_sz, stride=stride, transform=base_tf)

    predictions: Iterator['np.array'] = learner.predict_dataset(
        ds,
        raw_out=True,
        numpy_out=True,
        dataloader_kw=dict(batch_size=batch_sz),
        progress_bar=True,
        progress_bar_kw=dict(desc=f'Making predictions on {scene.id}'))

    labels = ChipClassificationLabels.from_predictions(ds.windows, predictions)

    return labels


def predict_scene_od(learner: 'ObjectDetectionLearner', scene: 'Scene',
                     predict_options: 'ObjectDetectionPredictOptions'
                     ) -> ObjectDetectionLabels:
    """Generate object detection predictions for a :class:`.Scene`."""
    from rastervision.pytorch_learner.dataset import (
        ObjectDetectionSlidingWindowGeoDataset)

    chip_sz = predict_options.chip_sz
    stride = predict_options.stride
    batch_sz = predict_options.batch_sz

    base_tf, _ = learner.cfg.data.get_data_transforms()
    ds = ObjectDetectionSlidingWindowGeoDataset(
        scene, size=chip_sz, stride=stride, transform=base_tf)

    predictions: Iterator[Dict[str, 'np.ndarray']] = learner.predict_dataset(
        ds,
        raw_out=True,
        numpy_out=True,
        predict_kw=dict(out_shape=(chip_sz, chip_sz)),
        dataloader_kw=dict(batch_size=batch_sz),
        progress_bar=True,
        progress_bar_kw=dict(desc=f'Making predictions on {scene.id}'))

    labels = ObjectDetectionLabels.from_predictions(ds.windows, predictions)
    labels = ObjectDetectionLabels.prune_duplicates(
        labels,
        score_thresh=predict_options.score_thresh,
        merge_thresh=predict_options.merge_thresh)

    return labels


def predict_scene_ss(learner: 'SemanticSegmentationLearner', scene: 'Scene',
                     predict_options: 'SemanticSegmentationPredictOptions'
                     ) -> 'SemanticSegmentationLabels':
    """Generate semantic segmentation predictions for a :class:`.Scene`."""
    from rastervision.pytorch_learner.dataset import (
        SemanticSegmentationSlidingWindowGeoDataset)

    if scene.label_store is None:
        raise ValueError(f'Scene.label_store is not set for scene {scene.id}')

    chip_sz = predict_options.chip_sz
    stride = predict_options.stride
    crop_sz = predict_options.crop_sz
    batch_sz = predict_options.batch_sz

    label_store: 'SemanticSegmentationLabelStore' = scene.label_store
    raw_out = label_store.smooth_output

    base_tf, _ = learner.cfg.data.get_data_transforms()
    pad_direction = 'end' if crop_sz is None else 'both'
    ds = SemanticSegmentationSlidingWindowGeoDataset(
        scene,
        size=chip_sz,
        stride=stride,
        pad_direction=pad_direction,
        transform=base_tf)

    predictions: Iterator['np.ndarray'] = learner.predict_dataset(
        ds,
        raw_out=raw_out,
        numpy_out=True,
        predict_kw=dict(out_shape=(chip_sz, chip_sz)),
        dataloader_kw=dict(batch_size=batch_sz),
        progress_bar=True,
        progress_bar_kw=dict(desc=f'Making predictions on {scene.id}'))

    labels = SemanticSegmentationLabels.from_predictions(
        ds.windows,
        predictions,
        smooth=raw_out,
        extent=scene.extent,
        num_classes=len(label_store.class_config),
        crop_sz=crop_sz)

    return labels
