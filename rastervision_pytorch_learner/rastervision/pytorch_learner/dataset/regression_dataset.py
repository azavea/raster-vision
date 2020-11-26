from typing import Iterable, Optional, Tuple
from os.path import join
import logging
import csv

import numpy as np
from PIL import Image
import albumentations as A

from torch.utils.data import Dataset

from rastervision.pytorch_learner.dataset import (ImageDataset, TransformType,
                                                  SlidingWindowGeoDataset,
                                                  RandomWindowGeoDataset)

log = logging.getLogger(__name__)


class RegressionDataReader(Dataset):
    def __init__(self, data_dir, class_names, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        labels_path = join(data_dir, 'labels.csv')
        with open(labels_path, 'r') as labels_file:
            labels_reader = csv.reader(labels_file, skipinitialspace=True)
            header = next(labels_reader)
            self.output_inds = [header.index(col) for col in class_names]
            self.labels = list(labels_reader)[1:]
        self.img_dir = join(data_dir, 'img')

    def __getitem__(self, ind) -> Tuple[np.ndarray, np.ndarray]:
        label_row = self.labels[ind]
        img_fn = label_row[0]

        y = [float(label_row[i]) for i in self.output_inds]
        y = np.array(y)
        img = Image.open(join(self.img_dir, img_fn))
        if self.transform:
            img = self.transform(img)
        return (img, y)

    def __len__(self):
        return len(self.labels)


class RegressionImageDataset(ImageDataset):
    def __init__(self,
                 data_dir: str,
                 class_names: Iterable[str],
                 transform: Optional[A.BasicTransform] = None):

        reader = RegressionDataReader(data_dir, class_names)
        super().__init__(
            orig_dataset=reader,
            transform=transform,
            transform_type=TransformType.regression)


class RegressionSlidingWindowGeoDataset(SlidingWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.regression)


class RegressionRandomWindowGeoDataset(RandomWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.regression)
