from os.path import join

from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

from rastervision.backend.torch_utils.data import DataBunch
from rastervision.backend.torch_utils.chip_classification.folder import (
    ImageFolder)


def build_databunch(data_dir, img_sz, batch_sz, class_names):
    num_workers = 4

    train_dir = join(data_dir, 'train')
    valid_dir = join(data_dir, 'valid')

    aug_transform = Compose([ToTensor()])
    transform = Compose([ToTensor()])

    train_ds = ImageFolder(train_dir, transform=aug_transform, classes=class_names)
    valid_ds = ImageFolder(valid_dir, transform=transform, classes=class_names)

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
