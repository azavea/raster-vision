from typing import Iterable
import logging

from rastervision.pytorch_learner.dataset import (
    ImageDataset, TransformType, SlidingWindowGeoDataset,
    RandomWindowGeoDataset, make_image_folder_dataset)

log = logging.getLogger(__name__)


class ClassificationImageDataset(ImageDataset):
    def __init__(self, data_dir: str, class_names: Iterable[str], *args,
                 **kwargs):
        ds = make_image_folder_dataset(data_dir, classes=class_names)
        super().__init__(
            ds, *args, **kwargs, transform_type=TransformType.classification)


class ClassificationSlidingWindowGeoDataset(SlidingWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.classification)

    def init_windows(self):
        super().init_windows()
        if self.scene.label_source is not None:
            self.scene.label_source.populate_labels(cells=self.windows)


class ClassificationRandomWindowGeoDataset(RandomWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.classification)
