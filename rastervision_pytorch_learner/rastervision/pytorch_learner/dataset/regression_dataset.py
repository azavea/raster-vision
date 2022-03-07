from typing import Iterable, Tuple
from os.path import join
import logging
import csv

import numpy as np
from torch.utils.data import Dataset

from rastervision.pytorch_learner.dataset import (
    ImageDataset, TransformType, SlidingWindowGeoDataset,
    RandomWindowGeoDataset, load_image)

log = logging.getLogger(__name__)


class RegressionDataReader(Dataset):
    def __init__(self, data_dir: str, class_names: Iterable[str]):
        self.data_dir = data_dir

        img_dir = join(data_dir, 'img')
        labels_path = join(data_dir, 'labels.csv')

        with open(labels_path, 'r') as labels_file:
            labels_reader = csv.reader(labels_file, skipinitialspace=True)
            all_rows = list(labels_reader)

        header, rows = all_rows[0], all_rows[1:]
        class_inds = [header.index(col) for col in class_names]
        self.targets = [[float(row[i]) for i in class_inds] for row in rows]
        self.img_paths = [join(img_dir, row[0]) for row in rows]

    def __getitem__(self, ind) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.img_paths[ind]
        targets = self.targets[ind]
        x = load_image(img_path)
        y = np.array(targets)
        return x, y

    def __len__(self):
        return len(self.labels)


class RegressionImageDataset(ImageDataset):
    def __init__(self, data_dir: str, class_names: Iterable[str], *args,
                 **kwargs):

        ds = RegressionDataReader(data_dir, class_names)
        super().__init__(
            ds, *args, **kwargs, transform_type=TransformType.regression)


class RegressionSlidingWindowGeoDataset(SlidingWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.regression)


class RegressionRandomWindowGeoDataset(RandomWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.regression)
