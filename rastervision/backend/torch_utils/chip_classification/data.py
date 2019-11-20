import logging
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda import device_count
from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import (
    Blur, RandomRotate90, HorizontalFlip, VerticalFlip, GaussianBlur,
    GaussNoise, RGBShift, ToGray)

from rastervision.backend.torch_utils.data import DataBunch
from rastervision.backend.torch_utils.chip_classification.folder import (
    ImageFolder)

log = logging.getLogger(__name__)


class AlbumentationDataset(Dataset):
    """An adapter to use arbitrary datasets with albumentations transforms."""

    def __init__(self, orig_dataset, transform=None):
        """Constructor.

        Args:
            orig_dataset: (Dataset) which is assumed to return PIL Image objects
                and not perform any transforms of its own
            transform: (albumentations.core.transforms_interface.ImageOnlyTransform)
        """
        self.orig_dataset = orig_dataset
        self.transform = transform

    def __getitem__(self, ind):
        x, y = self.orig_dataset[ind]
        x = np.array(x)
        if self.transform:
            x = self.transform(image=x)['image']
        x = torch.tensor(x).permute(2, 0, 1).float() / 255.0
        return x, y

    def __len__(self):
        return len(self.orig_dataset)


def build_databunch(data_dir, img_sz, batch_sz, class_names, augmentors):
    # To avoid memory errors
    if device_count() > 1:
        num_workers = 0
    else:
        num_workers = 4

    train_dir = join(data_dir, 'train')
    valid_dir = join(data_dir, 'valid')

    augmentors_dict = {
        'Blur': Blur(),
        'RandomRotate90': RandomRotate90(),
        'HorizontalFlip': HorizontalFlip(),
        'VerticalFlip': VerticalFlip(),
        'GaussianBlur': GaussianBlur(),
        'GaussNoise': GaussNoise(),
        'RGBShift': RGBShift(),
        'ToGray': ToGray()
    }

    aug_transforms = []
    for augmentor in augmentors:
        try:
            aug_transforms.append(augmentors_dict[augmentor])
        except KeyError as e:
            log.warning('{0} is an unknown augmentor. Continuing without {0}. \
                Known augmentors are: {1}'
                        .format(e, list(augmentors_dict.keys())))
    aug_transforms = Compose(aug_transforms)

    train_ds = AlbumentationDataset(
        ImageFolder(train_dir, classes=class_names), transform=aug_transforms)
    valid_ds = AlbumentationDataset(
        ImageFolder(valid_dir, classes=class_names))

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_sz,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True)

    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, class_names)
