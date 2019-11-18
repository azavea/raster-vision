from os.path import join

from torch import full
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from rastervision.backend.torch_utils.data import DataBunch
from rastervision.backend.torch_utils.chip_classification.folder import ImageFolder


def calculate_oversampling_weights(imageFolder, rare_classes, desired_prob):
    '''
    Calculates weights tensor for oversampling

    args:
        imageFolder: instance of
            rastervision.backend.torch_utils.chip_classification.folder.ImageFolder
        rare classes: (list) of ints of the classes that should be oversamples
        desired prob: (float) a single probability that the rare classes should
            have.
    returns:
        (tensor) with weights per index, e.g [0.5,0.1,0.9]
    '''

    chip_inds = []
    for rare_class_id in rare_classes:
        for idx, (sample, class_idx) in enumerate(imageFolder):
                if class_idx == rare_class_id:
                    chip_inds.append(idx)

    rare_weight = desired_prob / len(chip_inds)
    common_weight = (1.0 - desired_prob) / (len(imageFolder) - len(chip_inds))

    weights = full((len(imageFolder),), common_weight)
    weights[chip_inds] = rare_weight

    return weights


def build_databunch(data_dir, img_sz, batch_sz, class_names, rare_classes, desired_prob):
    num_workers = 4

    train_dir = join(data_dir, 'train')
    valid_dir = join(data_dir, 'valid')

    aug_transform = Compose([ToTensor()])
    transform = Compose([ToTensor()])

    train_ds = ImageFolder(
        train_dir, transform=aug_transform, classes=class_names)
    valid_ds = ImageFolder(valid_dir, transform=transform, classes=class_names)

    if rare_classes != []:
        train_sample_weights = calculate_oversampling_weights(train_ds, rare_classes,
                                                                desired_prob)
        valid_sample_weights = calculate_oversampling_weights(valid_ds, rare_classes,
                                                                desired_prob)

        def get_class_with_max_count(imageFolder):
            count_per_class = {}
            for class_idx in imageFolder.class_to_idx.values():
                count_per_class[class_idx] = 0
            for (sample_path, class_index) in imageFolder.samples:
                count_per_class[class_index] += 1
            largest_class_idx = max(count_per_class, key=count_per_class.get)
            return count_per_class[largest_class_idx]

        num_train_samples = len(train_ds.classes) * get_class_with_max_count(train_ds)
        num_valid_samples = len(valid_ds.classes) * get_class_with_max_count(valid_ds)

        train_sampler = WeightedRandomSampler(weights=train_sample_weights,
                                                num_samples=num_train_samples,
                                                replacement=True)
        valid_sampler = WeightedRandomSampler(weights=valid_sample_weights,
                                                num_samples=num_valid_samples,
                                                replacement=True)
    else:
        train_sampler = None
        valid_sampler = None

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_sz,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True,
        sampler=valid_sampler)

    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, class_names)
