from typing import TYPE_CHECKING
import logging

from rastervision.core.rv_pipeline import RVPipeline
from rastervision.core.data.label import ObjectDetectionLabels

if TYPE_CHECKING:
    from rastervision.core.data import Labels, Scene

log = logging.getLogger(__name__)


class ObjectDetection(RVPipeline):
    def predict_scene(self, scene: 'Scene') -> 'Labels':
        if self.backend is None:
            self.build_backend()

        # Use strided windowing to ensure that each object is fully visible (ie. not
        # cut off) within some window. This means prediction takes 4x longer for object
        # detection :(
        chip_sz = self.config.predict_chip_sz
        stride = chip_sz // 2
        labels = self.backend.predict_scene(
            scene, chip_sz=chip_sz, stride=stride)
        labels = self.post_process_predictions(labels, scene)
        return labels

    def post_process_predictions(self, labels: ObjectDetectionLabels,
                                 scene: 'Scene') -> ObjectDetectionLabels:
        return ObjectDetectionLabels.prune_duplicates(
            labels,
            score_thresh=self.config.predict_options.score_thresh,
            merge_thresh=self.config.predict_options.merge_thresh)
