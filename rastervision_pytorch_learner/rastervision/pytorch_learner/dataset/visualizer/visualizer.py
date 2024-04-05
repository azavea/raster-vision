from typing import (TYPE_CHECKING, Sequence, Optional, List, Dict, Union,
                    Tuple, Any)
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import albumentations as A
import matplotlib.pyplot as plt

from rastervision.pipeline.file_system import make_dir
from rastervision.core.data import ClassConfig
from rastervision.pytorch_learner.utils import (
    deserialize_albumentation_transform, validate_albumentation_transform,
    MinMaxNormalize)
from rastervision.pytorch_learner.learner_config import (
    RGBTuple,
    ChannelInds,
    validate_channel_display_groups,
    get_default_channel_display_groups,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from matplotlib.figure import Figure


class Visualizer(ABC):
    """Base class for plotting samples from computer vision PyTorch Datasets."""

    scale: float = 3.

    def __init__(self,
                 class_names: List[str],
                 class_colors: Optional[List[Union[str, RGBTuple]]] = None,
                 transform: Optional[Dict] = None,
                 channel_display_groups: Optional[Union[Dict[
                     str, ChannelInds], Sequence[ChannelInds]]] = None):
        """Constructor.

        Args:
            class_names: names of classes
            class_colors: Colors used to display classes. Can be color 3-tuples
                in list form.
            transform: An Albumentations transform serialized as a dict that
                will be applied to each image before it is plotted. Mainly useful
                for undoing any data transformation that you do not want included in
                the plot, such as normalization. The default value will shift and scale
                the image so the values range from 0.0 to 1.0 which is the expected range
                for the plotting function. This default is useful for cases where the
                values after normalization are close to zero which makes the plot
                difficult to see.
            channel_display_groups: Groups of image channels to display together as a
                subplot when plotting the data and predictions.
                Can be a list or tuple of groups (e.g. [(0, 1, 2), (3,)]) or a
                dict containing title-to-group mappings
                (e.g. {"RGB": [0, 1, 2], "IR": [3]}),
                where each group is a list or tuple of channel indices and
                title is a string that will be used as the title of the subplot
                for that group.
        """
        self.class_config = ClassConfig(names=class_names, colors=class_colors)
        if transform is None:
            transform = A.to_dict(MinMaxNormalize())
        self.transform = validate_albumentation_transform(transform)
        self._channel_display_groups = validate_channel_display_groups(
            channel_display_groups)

    @property
    def class_names(self):
        return self.class_config.names

    @property
    def class_colors(self):
        return self.class_config.colors

    @abstractmethod
    def plot_xyz(self,
                 axs,
                 x: Tensor,
                 y: Optional[Sequence] = None,
                 z: Optional[Sequence] = None,
                 plot_title: bool = True):
        """Plot image, ground truth labels, and predicted labels.

        Args:
            axs: matplotlib axes on which to plot
            x: image
            y: ground truth labels
            z: optional predicted labels
        """

    def plot_batch(self,
                   x: Tensor,
                   y: Optional[Sequence] = None,
                   output_path: Optional[str] = None,
                   z: Optional[Sequence] = None,
                   batch_limit: Optional[int] = None,
                   show: bool = False):
        """Plot a whole batch in a grid using plot_xyz.

        Args:
            x: batch of images
            y: ground truth labels
            output_path: local path where to save plot image
            z: optional predicted labels
            batch_limit: optional limit on (rendered) batch size
        """
        params = self.get_plot_params(
            x=x, y=y, z=z, output_path=output_path, batch_limit=batch_limit)
        if params['subplot_args']['nrows'] == 0:
            return

        if x.ndim == 4:
            fig, axs = plt.subplots(**params['fig_args'],
                                    **params['subplot_args'])
            plot_xyz_args = params['plot_xyz_args']
            self._plot_batch(fig, axs, plot_xyz_args, x, y=y, z=z)
        elif x.ndim == 5:
            # If a temporal dimension is present, we divide the figure into
            # multiple subfigures--one for each batch. Then, in each subfigure,
            # we plot all timesteps as if they were a single batch. To
            # delineate the boundary b/w batch items, we adopt the convention
            # of only displaying subplot titles once per batch (above the first
            # row in each batch).
            batch_sz, T, *_ = x.shape
            params['fig_args']['figsize'][1] *= T
            fig = plt.figure(**params['fig_args'])
            subfigs = fig.subfigures(
                nrows=batch_sz, ncols=1, hspace=0.0, squeeze=False)
            subfig_axs = [
                subfig.subplots(
                    nrows=T, ncols=params['subplot_args']['ncols'])
                for subfig in subfigs.flat
            ]
            for i, axs in enumerate(subfig_axs):
                plot_xyz_args = [
                    dict(params['plot_xyz_args'][i]) for _ in range(T)
                ]
                plot_xyz_args[0]['plot_title'] = True
                for args in plot_xyz_args[1:]:
                    args['plot_title'] = False
                _x = x[i]
                _y = None if y is None else [y[i]] * T
                _z = None if z is None else [z[i]] * T
                self._plot_batch(fig, axs, plot_xyz_args, _x, y=_y, z=_z)
        else:
            raise ValueError('Expected x to have 4 or 5 dims, but found '
                             f'x.shape: {x.shape}')

        if show:
            plt.show()
        if output_path is not None:
            make_dir(output_path, use_dirname=True)
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0.2)

        plt.close(fig)

    def _plot_batch(
            self,
            fig: 'Figure',
            axs: Sequence,
            plot_xyz_args: List[dict],
            x: Tensor,
            y: Optional[Sequence] = None,
            z: Optional[Sequence] = None,
    ):
        # (N, c, h, w) --> (N, h, w, c)
        x = x.permute(0, 2, 3, 1)

        # apply transform, if given
        if self.transform is not None:
            tf = deserialize_albumentation_transform(self.transform)
            imgs = [tf(image=img)['image'] for img in x.numpy()]
            x = torch.from_numpy(np.stack(imgs))

        if y is None:
            y = [None] * len(x)
        if z is None:
            z = [None] * len(x)
        for i, row_axs in enumerate(axs):
            self.plot_xyz(row_axs, x[i], y[i], z=z[i], **plot_xyz_args[i])

    def get_channel_display_groups(
            self, nb_img_channels: int
    ) -> Union[Dict[str, ChannelInds], Sequence[ChannelInds]]:
        # The default channel_display_groups object depends on the number of
        # channels in the image. This number is not known when the Visualizer
        # is constructed which is why it needs to be created later.
        if self._channel_display_groups is not None:
            return self._channel_display_groups
        return get_default_channel_display_groups(nb_img_channels)

    def get_collate_fn(self) -> Optional[callable]:
        """Returns a custom collate_fn to use in DataLoader.

        None is returned if default collate_fn should be used.

        See https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        """
        return None

    def get_batch(self, dataset: 'Dataset', batch_sz: int = 4,
                  **kwargs) -> Tuple[Tensor, Any]:
        """Generate a batch from a dataset.

        This is a convenience method for generating a batch of data to plot.

        Args:
            dataset (Dataset): A Pytorch Dataset.
            batch_sz (int): Batch size. Defaults to 4.
            **kwargs: Extra args for :class:`~torch.utils.data.DataLoader`.

        Returns:
            Tuple[Tensor, Any]: (x, y) tuple where x is images and y is labels.
        """
        collate_fn = self.get_collate_fn()
        dl = DataLoader(dataset, batch_sz, collate_fn=collate_fn, **kwargs)
        try:
            x, y = next(iter(dl))
        except StopIteration:
            raise ValueError('dataset did not return a batch')

        return x, y

    def get_plot_nrows(self, **kwargs) -> int:
        x = kwargs['x']
        batch_limit = kwargs.get('batch_limit')
        batch_sz = x.shape[0]
        nrows = min(batch_sz,
                    batch_limit) if batch_limit is not None else batch_sz
        return nrows

    def get_plot_ncols(self, **kwargs) -> int:
        x = kwargs['x']
        nb_img_channels = x.shape[1]
        ncols = len(self.get_channel_display_groups(nb_img_channels))
        return ncols

    def get_plot_params(self, **kwargs) -> dict:
        nrows = self.get_plot_nrows(**kwargs)
        ncols = self.get_plot_ncols(**kwargs)
        params = {
            'fig_args': {
                'constrained_layout': True,
                'figsize': np.array((self.scale * ncols, self.scale * nrows)),
            },
            'subplot_args': {
                'nrows': nrows,
                'ncols': ncols,
                'squeeze': False
            },
            'plot_xyz_args': [{} for _ in range(nrows)]
        }
        return params
