import logging
from os.path import join

from torch.utils.data import DataLoader
from torch import from_numpy
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import (
    Blur,
    Rotate,
    HorizontalFlip,
    GaussianBlur,
    GaussNoise)

from rastervision.backend.torch_utils.data import DataBunch
from rastervision.backend.torch_utils.chip_classification.folder import (
    ImageFolder)

log = logging.getLogger(__name__)


class ToTensor(ToTensorV2):
    def __init__(self, always_apply=True, p=1.0):
        super(ToTensorV2, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # Overrides default method because that returns a
        # torch.ByteTensor(), and we need a torch.FloatTensor().
        # because the model weights are torch.FloatTensor() as well
        return from_numpy(img.transpose(2, 0, 1)).float()


def build_databunch(data_dir, img_sz, batch_sz, class_names, augmentors):
    num_workers = 4

    train_dir = join(data_dir, 'train')
    valid_dir = join(data_dir, 'valid')

    augmentors_dict = {
        'Blur': Blur(),
        'Rotate': Rotate(),
        'HorizontalFlip': HorizontalFlip(),
        'GaussianBlur': GaussianBlur(),
        'GaussNoise': GaussNoise()}

    augmentors_placeholder = []
    for augmentor in augmentors:
        try:
            augmentors_placeholder.append(augmentors_dict[augmentor])
        except KeyError as e:
            log.warning('{0} is an unknown augmentor. Continuing without {0}. \
                Known augmentors are: {1}'.format(e, list(augmentors_dict.keys())))
    augmentors_placeholder.append(ToTensor())

    aug_transforms = Compose(augmentors_placeholder)
    transforms = Compose(augmentors_placeholder)

    train_ds = ImageFolder(
        train_dir,
        transform=aug_transforms,
        classes=class_names
    )

    valid_ds = ImageFolder(
        valid_dir,
        transform=transforms,
        classes=class_names
    )

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
