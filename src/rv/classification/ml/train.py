import json
import os
import shutil
import time
import csv
from collections import namedtuple

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import click
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import torchvision.datasets as datasets

from rv.classification.ml.utils import (
    AverageMeter, accuracy, ConfusionMeter, RandomVerticalFlip, RandomRotate90,
    Resize
)
from rv.utils import make_empty_dir

CUDA = torch.cuda.is_available()
print('Cuda: {}'.format(CUDA))
cudnn.benchmark = True


class CsvDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'images')
        self.data_rows = []
        self.transform = transform

        with open(os.path.join(data_dir, 'labels.csv')) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            for row in csv_reader:
                self.data_rows.append(row)

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, ind):
        filename, target = self.data_rows[ind]
        target = int(target)
        im = Image.open(os.path.join(self.image_dir, filename))
        im = self.transform(im)
        return (im, target)


def parse_config(config_path):
    with open(config_path) as config_file:
        config = json.load(
            config_file,
            object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        return config


def adjust_learning_rate(config, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = config.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_model(config):
    if config.pretrained:
        model = models.__dict__[config.arch](pretrained=True)
    else:
        model = models.__dict__[config.arch]()

    # modify network for nb_classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.num_classes)

    if CUDA:
        model = model.cuda()

    return model


def get_criterion(config):
    criterion = nn.CrossEntropyLoss()
    if CUDA:
        criterion.cuda()

    return criterion


def get_optimizer(config, model):
    optimizer = torch.optim.SGD(
        model.parameters(), config.lr, momentum=config.momentum,
        weight_decay=config.weight_decay)

    return optimizer


def load_checkpoint(checkpoint_path, model, optimizer):
    start_epoch = 0
    best_prec1 = 0
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_path, start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    return start_epoch, best_prec1


def get_train_loader(config, dataset_dir):
    transform = transforms.Compose([
        Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotate90(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])

    if config.dataset == 'folder':
        dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    elif config.dataset == 'csv':
        dataset = CsvDataset(dataset_dir, transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.workers)

    return loader


def get_val_loader(config, dataset_dir):
    transform = transforms.Compose([
        Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])

    if config.dataset == 'folder':
        dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    elif config.dataset == 'csv':
        dataset = CsvDataset(dataset_dir, transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers)

    return loader


def train_epoch(config, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topK = AverageMeter()
    confusion = ConfusionMeter(config.num_classes, normalized=True)

    # switch to train mode
    model.train()

    end = time.time()
    for batch_ind, (input, target) in enumerate(train_loader):
        if CUDA:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        data_time.update(time.time() - end)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, precK = accuracy(output.data, target, topK=config.topK)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        topK.update(precK[0], input.size(0))
        confusion.add(output.data, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_ind % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@K {topK.val:.3f} ({topK.avg:.3f})'.format(
                   epoch, batch_ind, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, topK=topK))
            #print('Confusion: {}'.format(confusion.value()))


def validate(config, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topK = AverageMeter()
    confusion = ConfusionMeter(config.num_classes, normalized=True)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_ind, (input, target) in enumerate(val_loader):
        if CUDA:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, precK = accuracy(output.data, target, topK=config.topK)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        topK.update(precK[0], input.size(0))
        confusion.add(output.data, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_ind % config.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@K {topK.val:.3f} ({topK.avg:.3f})'.format(
                   batch_ind, len(val_loader), batch_time=batch_time,
                   loss=losses, top1=top1, topK=topK))

    print(' * Prec@1 {top1.avg:.3f} Prec@K {topK.avg:.3f}'
          .format(top1=top1, topK=topK))
    print('Confusion: {}'.format(confusion.value()))

    return top1.avg


def _train(config_path, train_dataset_dir, val_dataset_dir, output_dir):
    config = parse_config(config_path)
    model = get_model(config)
    criterion = get_criterion(config)
    optimizer = get_optimizer(config, model)

    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth.tar')
    best_checkpoint_path = os.path.join(
        output_dir, 'best-checkpoint.pth.tar')
    start_epoch, best_prec1 = load_checkpoint(
        checkpoint_path, model, optimizer)

    train_loader = get_train_loader(config, train_dataset_dir)
    val_loader = get_val_loader(config, val_dataset_dir)

    print('Starting training...')
    make_empty_dir(output_dir, empty_dir=False)
    for epoch in range(start_epoch, config.epochs):
        adjust_learning_rate(config, optimizer, epoch)
        train_epoch(config, train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(config, val_loader, model, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        state = {
            'epoch': epoch + 1,
            'arch': config.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, best_checkpoint_path)


@click.command()
@click.argument('config_path')
@click.argument('train_dataset_dir')
@click.argument('val_dataset_dir')
@click.argument('output_dir')
def train(config_path, train_dataset_dir, val_dataset_dir, output_dir):
    _train(config_path, train_dataset_dir, val_dataset_dir, output_dir)


if __name__ == '__main__':
    train()
