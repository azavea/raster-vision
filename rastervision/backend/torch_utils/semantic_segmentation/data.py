from os.path import join, basename
import glob

from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset

from rastervision.backend.torch_utils.data import DataBunch


class ToTensor(object):
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, x, y):
        return (self.to_tensor(x), (255 * self.to_tensor(y)).squeeze().long())


class ComposeTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.img_paths = glob.glob(join(data_dir, 'img', '*.png'))
        self.transforms = transforms

    def __getitem__(self, ind):
        img_path = self.img_paths[ind]
        label_path = join(self.data_dir, 'labels', basename(img_path))
        x = Image.open(img_path)
        y = Image.open(label_path)

        if self.transforms is not None:
            x, y = self.transforms(x, y)
        return (x, y)

    def __len__(self):
        return len(self.img_paths)


def build_databunch(data_dir, img_sz, batch_sz, class_names):
    num_workers = 4

    train_dir = join(data_dir, 'train')
    valid_dir = join(data_dir, 'valid')

    aug_transforms = ComposeTransforms([ToTensor()])
    transforms = ComposeTransforms([ToTensor()])

    train_ds = SegmentationDataset(train_dir, transforms=aug_transforms)
    valid_ds = SegmentationDataset(valid_dir, transforms=transforms)

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
