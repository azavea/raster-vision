from typing import Tuple
from pathlib import Path

import logging

import numpy as np
from torch.utils.data import Dataset

from rastervision.pytorch_learner.dataset import (
    ImageDataset, TransformType, SlidingWindowGeoDataset,
    RandomWindowGeoDataset, load_image, discover_images, ImageDatasetError)

log = logging.getLogger(__name__)


class SemanticSegmentationDataReader(Dataset):
    """Reads semantic segmentatioin images and labels from files."""

    def __init__(self, data_dir: str):
        """Constructor.

        data_dir is assumed to have an 'img' subfolder that contains image
        files and a 'labels' subfolder that contains label files.

        Args:
            data_dir (str): Root directory that contains image and label files.
        """
        self.data_dir = Path(data_dir)
        img_dir = self.data_dir / 'img'
        label_dir = self.data_dir / 'labels'

        # collect image and label paths, match them based on filename
        img_paths = discover_images(img_dir)
        label_paths = discover_images(label_dir)
        self.img_paths = sorted(img_paths, key=lambda p: p.stem)
        self.label_paths = sorted(label_paths, key=lambda p: p.stem)
        self.validate_paths()

    def validate_paths(self) -> None:
        if len(self.img_paths) != len(self.label_paths):
            raise ImageDatasetError(
                'There should be a label file for every image file. '
                f'Found {len(self.img_paths)} image files and '
                f'{len(self.label_paths)} label files.')
        for img_path, label_path in zip(self.img_paths, self.label_paths):
            if img_path.stem != label_path.stem:
                raise ImageDatasetError(
                    f'Name mismatch between image file {img_path.stem} '
                    f'and label file {label_path.stem}.')

    def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.img_paths[ind]
        label_path = self.label_paths[ind]

        x = load_image(img_path)
        y = load_image(label_path).squeeze()

        return x, y

    def __len__(self):
        return len(self.img_paths)


class SemanticSegmentationImageDataset(ImageDataset):
    def __init__(self, data_dir: str, *args, **kwargs):

        ds = SemanticSegmentationDataReader(data_dir)
        super().__init__(
            ds,
            *args,
            **kwargs,
            transform_type=TransformType.semantic_segmentation)


class SemanticSegmentationSlidingWindowGeoDataset(SlidingWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            transform_type=TransformType.semantic_segmentation)


class SemanticSegmentationRandomWindowGeoDataset(RandomWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            transform_type=TransformType.semantic_segmentation)
