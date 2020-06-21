from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import ImageColor


def color_to_triple(color: Optional[str] = None) -> Tuple[int, int, int]:
    """Given a PIL ImageColor string, return a triple of integers
    representing the red, green, and blue values.

    If color is None, return a random color.

    Args:
         color: A PIL ImageColor string

    Returns:
         An triple of integers

    """
    if color is None:
        r = np.random.randint(0, 0x100)
        g = np.random.randint(0, 0x100)
        b = np.random.randint(0, 0x100)
        return (r, g, b)
    else:
        return ImageColor.getrgb(color)


def compute_conf_mat(out, y, num_labels):
    labels = torch.arange(0, num_labels).to(out.device)
    return ((out == labels[:, None]) & (y == labels[:, None, None])).sum(
        dim=2, dtype=torch.float32)


def compute_conf_mat_metrics(conf_mat, label_names, eps=1e-6):
    # eps is to avoid dividing by zero.
    eps = torch.tensor(eps)
    conf_mat = conf_mat.cpu()
    gt_count = conf_mat.sum(dim=1)
    pred_count = conf_mat.sum(dim=0)
    total = conf_mat.sum()
    true_pos = torch.diag(conf_mat)
    precision = true_pos / torch.max(pred_count, eps)
    recall = true_pos / torch.max(gt_count, eps)
    f1 = (2 * precision * recall) / torch.max(precision + recall, eps)

    weights = gt_count / total
    weighted_precision = (weights * precision).sum()
    weighted_recall = (weights * recall).sum()
    weighted_f1 = ((2 * weighted_precision * weighted_recall) / torch.max(
        weighted_precision + weighted_recall, eps))

    metrics = {
        'avg_precision': weighted_precision.item(),
        'avg_recall': weighted_recall.item(),
        'avg_f1': weighted_f1.item()
    }
    for ind, label in enumerate(label_names):
        metrics.update({
            '{}_precision'.format(label): precision[ind].item(),
            '{}_recall'.format(label): recall[ind].item(),
            '{}_f1'.format(label): f1[ind].item(),
        })
    return metrics


class AlbumentationsDataset(Dataset):
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
