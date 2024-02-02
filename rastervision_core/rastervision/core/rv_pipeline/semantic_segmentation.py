from typing import TYPE_CHECKING
import logging

import numpy as np

from rastervision.core.rv_pipeline import RVPipeline

if TYPE_CHECKING:
    from rastervision.core.data import (
        Labels,
        Scene,
    )
    from rastervision.core.rv_pipeline.semantic_segmentation_config import (
        SemanticSegmentationConfig)

log = logging.getLogger(__name__)


class SemanticSegmentation(RVPipeline):
    def post_process_batch(self, windows, chips, labels):
        # Fill in null class for any NODATA pixels.
        null_class_id = self.config.dataset.class_config.null_class_id
        for window, chip in zip(windows, chips):
            nodata_mask = np.sum(chip, axis=2) == 0
            labels.mask_fill(window, nodata_mask, fill_value=null_class_id)

        return labels

    def predict_scene(self, scene: 'Scene') -> 'Labels':
        if self.backend is None:
            self.build_backend()

        cfg: 'SemanticSegmentationConfig' = self.config
        chip_sz = cfg.predict_chip_sz
        stride = cfg.predict_options.stride
        crop_sz = cfg.predict_options.crop_sz

        if stride is None:
            stride = chip_sz

        if crop_sz == 'auto':
            overlap_sz = chip_sz - stride
            if overlap_sz % 2 == 1:
                log.warning(
                    'Using crop_sz="auto" but overlap size (chip_sz minus '
                    'stride) is odd. This means that one pixel row/col will '
                    'still overlap after cropping.')
            crop_sz = overlap_sz // 2

        labels = self.backend.predict_scene(
            scene, chip_sz=chip_sz, stride=stride, crop_sz=crop_sz)
        labels = self.post_process_predictions(labels, scene)
        return labels
