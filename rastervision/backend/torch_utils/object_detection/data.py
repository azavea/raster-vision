from os.path import join
from collections import defaultdict
import random
import glob

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision

from rastervision.utils.files import (file_to_json)
from rastervision.backend.torch_utils.object_detection.boxlist import BoxList
from rastervision.backend.torch_utils.data import DataBunch


class ToTensor(object):
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, x, y):
        return (self.to_tensor(x), y)


class ScaleTransform(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, x, y):
        yscale = self.height / x.shape[1]
        xscale = self.width / x.shape[2]
        x = F.interpolate(
            x.unsqueeze(0), size=(self.height, self.width), mode='bilinear')[0]
        return (x, y.scale(yscale, xscale))


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, x, y):
        if random.random() < self.prob:
            height, width = x.shape[-2:]
            x = x.flip(-1)

            boxes = y.boxes
            boxes[:, [1, 3]] = width - boxes[:, [3, 1]]
            y.boxes = boxes

        return (x, y)


class ComposeTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


def collate_fn(data):
    x = [d[0].unsqueeze(0) for d in data]
    y = [d[1] for d in data]
    return (torch.cat(x), y)


class CocoDataset(Dataset):
    def __init__(self, img_dir, annotation_uris, transforms=None):
        self.img_dir = img_dir
        self.annotation_uris = annotation_uris
        self.transforms = transforms

        self.imgs = []
        self.img2id = {}
        self.id2img = {}
        self.id2boxes = defaultdict(lambda: [])
        self.id2labels = defaultdict(lambda: [])
        self.label2name = {}
        for annotation_uri in annotation_uris:
            ann_json = file_to_json(annotation_uri)
            for img in ann_json['images']:
                self.imgs.append(img['file_name'])
                self.img2id[img['file_name']] = img['id']
                self.id2img[img['id']] = img['file_name']
            for ann in ann_json['annotations']:
                img_id = ann['image_id']
                box = ann['bbox']
                label = ann['category_id']
                box = torch.tensor(
                    [[box[1], box[0], box[1] + box[3], box[0] + box[2]]])
                self.id2boxes[img_id].append(box)
                self.id2labels[img_id].append(label)
        self.id2boxes = dict([(id, torch.cat(boxes).float())
                              for id, boxes in self.id2boxes.items()])
        self.id2labels = dict([(id, torch.tensor(labels))
                               for id, labels in self.id2labels.items()])

    def __getitem__(self, ind):
        img_fn = self.imgs[ind]
        img_id = self.img2id[img_fn]
        img = Image.open(join(self.img_dir, img_fn))

        if img_id in self.id2boxes:
            boxes, labels = self.id2boxes[img_id], self.id2labels[img_id]
            boxlist = BoxList(boxes, labels=labels)
        else:
            boxlist = BoxList(
                torch.empty((0, 4)), labels=torch.empty((0, )).long())
        if self.transforms:
            return self.transforms(img, boxlist)
        return (img, boxlist)

    def __len__(self):
        return len(self.imgs)


def get_label_names(coco_path):
    categories = file_to_json(coco_path)['categories']
    label2name = dict([(cat['id'], cat['name']) for cat in categories])
    labels = ['background'
              ] + [label2name[i] for i in range(1,
                                                len(label2name) + 1)]
    return labels


def build_databunch(data_dir, img_sz, batch_sz):
    # TODO This is to avoid freezing in the middle of the first epoch. Would be nice
    # to fix this.
    num_workers = 0

    train_dir = join(data_dir, 'train')
    train_anns = glob.glob(join(train_dir, '*.json'))
    valid_dir = join(data_dir, 'valid')
    valid_anns = glob.glob(join(valid_dir, '*.json'))

    label_names = get_label_names(train_anns[0])
    aug_transforms = ComposeTransforms(
        [ToTensor(),
         ScaleTransform(img_sz, img_sz),
         RandomHorizontalFlip()])
    transforms = ComposeTransforms(
        [ToTensor(), ScaleTransform(img_sz, img_sz)])

    train_ds = CocoDataset(train_dir, train_anns, transforms=aug_transforms)
    valid_ds = CocoDataset(valid_dir, valid_anns, transforms=transforms)
    train_ds.label_names = label_names
    valid_ds.label_names = label_names

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True)
    valid_dl = DataLoader(
        valid_ds,
        collate_fn=collate_fn,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True)
    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, label_names)
