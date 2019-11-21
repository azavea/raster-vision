import logging
from os.path import join

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import (
    Blur, RandomRotate90, HorizontalFlip, VerticalFlip, GaussianBlur,
    GaussNoise, RGBShift, ToGray)

from rastervision.backend.torch_utils.data import DataBunch
from rastervision.backend.torch_utils.chip_classification.folder import ImageFolder

log = logging.getLogger(__name__)


def calculate_oversampling_weights(targets, rare_classes, desired_prob):
    '''
    Calculates weights tensor for oversampling

    args:
        targets: (list) containing the class labels/targets/y-data
        rare classes: (list) of ints of the class labels that should be oversamples
        desired prob: (float) a single probability that the rare classes should
            have.
    returns:
        (tensor) with weights per index, e.g [0.5,0.1,0.9]
    '''

    chip_idx = []
    for idx, target in enumerate(targets):
        if target in rare_classes:
            chip_idx.append(idx)

    rare_weight = desired_prob / len(chip_idx)
    common_weight = (1.0 - desired_prob) / (len(targets) - len(chip_idx))

    weights = torch.full((len(targets), ), common_weight)
    weights[chip_idx] = rare_weight

    return weights


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


def build_databunch(data_dir, img_sz, batch_sz, class_names, rare_classes,
                    desired_prob, augmentors):
    num_workers = 0

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

    if rare_classes != []:
        targets = [target[1] for target in train_ds.orig_dataset.imgs]
        train_sample_weights = calculate_oversampling_weights(
            targets, rare_classes, desired_prob)
        num_train_samples = len(train_ds)
        train_sampler = WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=num_train_samples,
            replacement=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dl = DataLoader(
        train_ds,
        shuffle=shuffle,
        batch_size=batch_sz,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True)

    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, class_names)
