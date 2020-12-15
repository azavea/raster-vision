import logging

from rastervision.pytorch_learner.dataset import (ImageDataset, TransformType,
                                                  SlidingWindowGeoDataset,
                                                  RandomWindowGeoDataset)

log = logging.getLogger(__name__)


class ClassificationImageDataset(ImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.classification)


class ClassificationSlidingWindowGeoDataset(SlidingWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.classification)

    def init_windows(self):
        super().init_windows()
        self.scene.label_source.populate_labels(cells=self.windows)


class ClassificationRandomWindowGeoDataset(RandomWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.classification)
