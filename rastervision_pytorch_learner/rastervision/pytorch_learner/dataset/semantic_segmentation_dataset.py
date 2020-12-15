from typing import Tuple, Optional
from pathlib import Path

import logging

import numpy as np
from PIL import Image
import albumentations as A

from rastervision.pytorch_learner.dataset import (ImageDataset, TransformType,
                                                  SlidingWindowGeoDataset,
                                                  RandomWindowGeoDataset)

log = logging.getLogger(__name__)


class SemanticSegmentationDataReader():
    """Reads semantic segmentatioin images and labels from files."""

    def __init__(self,
                 data_dir: str,
                 img_fmt: str = 'png',
                 label_fmt: str = 'png'):
        """Constructor.

        data_dir is assumed to have an 'img' subfolder that contains image
        files and a 'labels' subfolder that contains label files. img_fmt and
        label_fmt specify the file format of image and label files
        respectively.

        Args:
            data_dir (str): Root directory that contains image and label files.
            img_fmt (str, optional): File format of the image files.
                Defaults to 'png'.
            label_fmt (str, optional): File format of the label files.
                Defaults to 'png'.
        """
        self.data_dir = Path(data_dir)
        img_dir = self.data_dir / 'img'
        label_dir = self.data_dir / 'labels'

        # collect image and label paths
        self.img_paths = list(img_dir.glob(f'*.{img_fmt}'))
        self.label_paths = [
            label_dir / f'{p.stem}.{label_fmt}' for p in self.img_paths
        ]

        # choose image loading method based on format
        if img_fmt.lower() in ('npy', 'npz'):
            self.img_load_fn = np.load
        else:
            self.img_load_fn = lambda path: np.array(Image.open(path))

        # choose label loading method based on format
        if label_fmt.lower() in ('npy', 'npz'):
            self.label_load_fn = np.load
        else:
            self.label_load_fn = lambda path: np.array(Image.open(path))

    def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.img_paths[ind]
        label_path = self.label_paths[ind]

        x = self.img_load_fn(img_path)
        y = self.label_load_fn(label_path)

        if x.ndim == 2:
            # (h, w) --> (h, w, 1)
            x = x[..., np.newaxis]

        return (x, y)

    def __len__(self):
        return len(self.img_paths)


class SemanticSegmentationImageDataset(ImageDataset):
    def __init__(self,
                 data_dir: str,
                 img_fmt: str = 'png',
                 label_fmt: str = 'png',
                 transform: Optional[A.BasicTransform] = None):

        reader = SemanticSegmentationDataReader(data_dir, img_fmt, label_fmt)
        super().__init__(
            orig_dataset=reader,
            transform=transform,
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
