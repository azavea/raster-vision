import random

import numpy as np
from PIL import Image


class Meter(object):
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


# Taken from https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
class ConfusionMeter(Meter):
    """
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.
    """

    def __init__(self, k, normalized=False):
        """
        Args:
            k (int): number of classes in the classification problem
            normalized (boolean): Determines whether or not the confusion matrix
                is normalized or not
        """
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """
        Computes the confusion matrix of K x K size where K is no of classes
        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 1 and K.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 1 and K or N x K tensor, where targets are
                assumed to be provided as one-hot vectors
        """
        predicted = predicted.cpu().squeeze().numpy()
        target = target.cpu().squeeze().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5"""

    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotate90(object):
    """Randomly rotate the PIL image in increments of 90 degrees."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Randomly rotated image.
        """
        rot_choices = [
            None, Image.ROTATE_90, Image.ROTATE_180,
            Image.ROTATE_270]
        rot_choice = random.choice(rot_choices)
        if rot_choice is None:
            return img
        return img.transpose(rot_choice)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return img.resize(self.size)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topK=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxK = max(topK)
    batch_size = target.size(0)

    _, pred = output.topk(maxK, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topK:
        correctK = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correctK.mul_(100.0 / batch_size))
    return res
