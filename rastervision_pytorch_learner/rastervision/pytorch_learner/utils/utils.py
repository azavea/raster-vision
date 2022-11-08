from typing import Dict, Sequence, Tuple, Optional, Union, List, Iterable
from os.path import basename, join
import logging

import torch
from torch import nn
from torch.hub import _import_module
import numpy as np
from PIL import ImageColor
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.pipeline.config import ConfigError

log = logging.getLogger(__name__)


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


def validate_albumentation_transform(tf_dict: Optional[dict]) -> dict:
    """ Validate a serialized albumentation transform by attempting to
    deserialize it.
    """
    if tf_dict is not None:
        try:
            lambda_transforms_path = tf_dict.get('lambda_transforms_path',
                                                 None)
            # hack: if this is being called while building the config from the
            # bundle, skip the validation because the 'lambda_transforms_path's
            # have not been adjusted yet
            if (lambda_transforms_path is not None
                    and lambda_transforms_path.startswith('model-bundle')):
                return tf_dict
            else:
                _ = deserialize_albumentation_transform(tf_dict)
        except Exception:
            raise ConfigError('The given serialization is invalid. Use '
                              'A.to_dict(transform) to serialize.')
    return tf_dict


def serialize_albumentation_transform(
        tf: A.BasicTransform,
        lambda_transforms_path: Optional[str] = None,
        dst_dir: Optional[str] = None) -> dict:
    """Serialize an albumentations transform to a dict.

    If the transform includes a Lambda transform, a `lambda_transforms_path`
    should be provided. This should be a path to a python file that defines a
    dict named `lambda_transforms` as required by `A.from_dict()`. See
    https://albumentations.ai/docs/examples/serialization/ for details. This
    path is saved as a field in the returned dict so that it is available
    at the time of deserialization.

    Args:
        tf (A.BasicTransform): The transform to serialize.
        lambda_transforms_path (Optional[str], optional): Path to a python file
            that defines a dict named `lambda_transforms` as required by
            `A.from_dict()`. Defaults to None.
        dst_dir (Optional[str], optional): Directory to copy the transforms
            file to. Useful for copying the file to S3 when running on Batch.
            Defaults to None.

    Returns:
        dict: The serialized transform.
    """
    tf_dict = A.to_dict(tf)

    if lambda_transforms_path is not None:
        if dst_dir is not None:
            from rastervision.pipeline.file_system import upload_or_copy

            filename = basename(lambda_transforms_path)
            dst_uri = join(dst_dir, filename)
            upload_or_copy(lambda_transforms_path, dst_uri)
            lambda_transforms_path = dst_uri
        # save the path in the dict so that it is available
        # at deserialization time
        tf_dict['lambda_transforms_path'] = lambda_transforms_path

    return tf_dict


def deserialize_albumentation_transform(tf_dict: dict) -> A.BasicTransform:
    """Deserialize an albumentations transform serialized by
    `serialize_albumentation_transform()`.

    If the input dict contains a `lambda_transforms_path`, the
    `lambda_transforms` dict is dynamically imported from it and passed to
    `A.from_dict()`. See
    https://albumentations.ai/docs/examples/serialization/ for details

    Args:
        tf_dict (dict): Serialized albumentations transform.

    Returns:
        A.BasicTransform: Deserialized transform.
    """
    lambda_transforms_path = tf_dict.get('lambda_transforms_path', None)
    if lambda_transforms_path is not None:
        from rastervision.pipeline.file_system import download_if_needed

        with get_tmp_dir() as tmp_dir:
            filename = basename(lambda_transforms_path)
            # download the transforms definition file into tmp_dir
            lambda_transforms_path = download_if_needed(
                lambda_transforms_path, tmp_dir)
            # import it as a module
            lambda_transforms_module = _import_module(
                name=filename, path=lambda_transforms_path)
            # retrieve the lambda_transforms dict from the module
            lambda_transforms: dict = getattr(lambda_transforms_module,
                                              'lambda_transforms')
            # de-serialize
            tf = A.from_dict(tf_dict, nonserializable=lambda_transforms)
    else:
        tf = A.from_dict(tf_dict)
    return tf


class SplitTensor(nn.Module):
    """ Wrapper around `torch.split` """

    def __init__(self, size_or_sizes, dim):
        super().__init__()
        self.size_or_sizes = size_or_sizes
        self.dim = dim

    def forward(self, X):
        return X.split(self.size_or_sizes, dim=self.dim)


class Parallel(nn.ModuleList):
    """ Passes inputs through multiple `nn.Module`s in parallel.
        Returns a tuple of outputs.
    """

    def __init__(self, *args):
        super().__init__(args)

    def forward(self, xs):
        if isinstance(xs, torch.Tensor):
            return tuple(m(xs) for m in self)
        assert len(xs) == len(self)
        return tuple(m(x) for m, x in zip(self, xs))


class AddTensors(nn.Module):
    """ Adds all its inputs together. """

    def forward(self, xs):
        return sum(xs)


class MinMaxNormalize(ImageOnlyTransform):
    """Albumentations transform that normalizes image to desired min and max values.

    This will shift and scale the image appropriately to achieve the desired min and
    max.
    """

    def __init__(
            self,
            min_val=0.0,
            max_val=1.0,
            dtype=cv2.CV_32F,
            always_apply=False,
            p=1.0,
    ):
        """Constructor.

        Args:
            min_val: the minimum value that output should have
            max_val: the maximum value that output should have
            dtype: the dtype of output image
        """
        super(MinMaxNormalize, self).__init__(always_apply, p)
        self.min_val = min_val
        self.max_val = max_val
        self.dtype = dtype

    def _apply_on_channel(self, image, **params):
        out = cv2.normalize(
            image,
            None,
            self.min_val,
            self.max_val,
            cv2.NORM_MINMAX,
            dtype=self.dtype)
        # We need to clip because sometimes values are slightly less or more than
        # min_val and max_val due to rounding errors.
        return np.clip(out, self.min_val, self.max_val)

    def apply(self, image, **params):
        if image.ndim <= 2:
            return self._apply_on_channel(image, **params)

        assert image.ndim == 3

        chs = [
            self._apply_on_channel(ch, **params)
            for ch in image.transpose(2, 0, 1)
        ]
        out = np.stack(chs, axis=2)
        return out

    def get_transform_init_args_names(self):
        return ('min_val', 'max_val', 'dtype')


def adjust_conv_channels(old_conv: nn.Conv2d,
                         in_channels: int,
                         pretrained: bool = True
                         ) -> Union[nn.Conv2d, nn.Sequential]:

    if in_channels == old_conv.in_channels:
        return old_conv

    # These parameters will be the same for the new conv layer.
    # This list should be kept up to date with the Conv2d definition.
    old_conv_args = {
        'out_channels': old_conv.out_channels,
        'kernel_size': old_conv.kernel_size,
        'stride': old_conv.stride,
        'padding': old_conv.padding,
        'dilation': old_conv.dilation,
        'groups': old_conv.groups,
        'bias': old_conv.bias is not None,
        'padding_mode': old_conv.padding_mode
    }

    if not pretrained:
        # simply replace the first conv layer with one with the
        # correct number of input channels
        new_conv = nn.Conv2d(in_channels=in_channels, **old_conv_args)
        return new_conv

    if in_channels > old_conv.in_channels:
        # insert a new conv layer parallel to the existing one
        # and sum their outputs
        extra_channels = in_channels - old_conv.in_channels
        extra_conv = nn.Conv2d(in_channels=extra_channels, **old_conv_args)
        new_conv = nn.Sequential(
            # split input along channel dim
            SplitTensor((old_conv.in_channels, extra_channels), dim=1),
            # each split goes to its respective conv layer
            Parallel(old_conv, extra_conv),
            # sum the parallel outputs
            AddTensors())
        return new_conv
    elif in_channels < old_conv.in_channels:
        new_conv = nn.Conv2d(in_channels=in_channels, **old_conv_args)
        pretrained_kernels = old_conv.weight.data[:, :in_channels]
        new_conv.weight.data[:, :in_channels] = pretrained_kernels
        return new_conv
    else:
        raise ConfigError(f'Something went wrong.')


def plot_channel_groups(axs: Iterable,
                        imgs: Iterable[Union[np.array, torch.Tensor]],
                        channel_groups: dict) -> None:
    for title, ax, img in zip(channel_groups.keys(), axs, imgs):
        ax.imshow(img)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])


def channel_groups_to_imgs(
        x: torch.Tensor,
        channel_groups: Dict[str, Sequence[int]]) -> List[torch.Tensor]:
    imgs = []
    for title, ch_inds in channel_groups.items():
        img = x[..., ch_inds]
        if len(ch_inds) == 1:
            # repeat single channel 3 times
            img = img.expand(-1, -1, 3)
        elif len(ch_inds) == 2:
            # add a 3rd channel with all pixels set to 0.5
            h, w, _ = x.shape
            third_channel = torch.full((h, w, 1), fill_value=.5)
            img = torch.cat((img, third_channel), dim=-1)
        elif len(ch_inds) > 3:
            # only use the first 3 channels
            log.warn(f'Only plotting first 3 channels of channel-group '
                     f'{title}: {ch_inds}.')
            img = x[..., ch_inds[:3]]
        imgs.append(img)
    return imgs
