from os.path import join

import logging
log = logging.getLogger(__name__)

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Transpose,
    Rotate,
    GaussNoise,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    ChannelShuffle,
    InvertImg,
    ToGray,
    RandomSnow,
    RandomFog,
    ChannelDropout
)

from rastervision.backend.torch_utils.data import DataBunch

def build_databunch(data_dir, img_sz, batch_sz, class_names, augmentors):
    num_workers = 4

    train_dir = join(data_dir, 'train')
    valid_dir = join(data_dir, 'valid')

    aug_transform = [ToTensor()]
    transform = [ToTensor()]

    for augmentor in augmentors:
        if augmentor == 'RandomHorizontalFlip':
            aug_transform.append(HorizontalFlip(p=0.5))
        elif augmentor == 'RandomVerticalFlip':
            aug_transform.append(VerticalFlip(p=0.5))
        elif augmentor == '360Rotation':
            aug_transform.append(Rotate(p=1.0,limit=360))
        elif augmentor == 'RandomGaussianNoise':
            aug_transform.append(GaussNoise(p=0.5))
        elif augmentor == 'RandomGamma':
            aug_transform.append(RandomGamma(p=0.5))
        elif augmentor == 'HueSaturationValue':
            aug_transform.append(HueSaturationValue(p=0.5))
        elif augmentor == 'RGBShift':
            aug_transform.append(RGBShift(p=0.5))
        elif augmentor == 'RandomBrightness':
            aug_transform.append(RandomBrightness(p=0.5))
        elif augmentor == 'RandomContrast':
            aug_transform.append(RandomContrast(p=0.5))
        elif augmentor == 'ChannelShuffle':
            aug_transform.append(ChannelShuffle(p=0.5))
        elif augmentor == 'InvertImg':
            aug_transform.append(InvertImg(p=0.5))
        elif augmentor == 'ToGray':
            aug_transform.append(ToGray(p=0.5))
        elif augmentor == 'RandomSnow':
            aug_transform.append(RandomSnow(p=0.5))
        elif augmentor == 'RandomFog':
            aug_transform.append(RandomFog(p=0.5))
        elif augmentor == 'ChannelDropout':
            aug_transform.append(ChannelDropout(p=0.5))
        else:
            log.warning('Unknown augmentor: {}, is the spelling correct?'.format(augmentor))

    aug_transform = Compose(aug_transform)
    transform = Compose(transform)

    train_ds = ImageFolder(train_dir, transform=aug_transform)
    valid_ds = ImageFolder(valid_dir, transform=transform)

    class_to_idx = dict(
        [(class_name, idx) for idx, class_name in enumerate(class_names)])
    train_ds.classes = class_names
    train_ds.class_to_idx = class_to_idx
    valid_ds.classes = class_names
    valid_ds.class_to_idx = class_to_idx

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True)

    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, class_names)