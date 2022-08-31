from typing import (Sequence, Optional, List, Dict, Union, Tuple, NonNegInt)
from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor
import torch
import matplotlib.pyplot as plt

from rastervision.pipeline.file_system import make_dir
from rastervision.pytorch_learner.utils import (
    deserialize_albumentation_transform)

RGBTuple = Tuple[int, int, int]


class Visualizer():
    def __init__(self, 
                 class_names: Optional[List[str]] = None, 
                 class_colors: Optional[List[Union[str, RGBTuple]]] = None,
                 transform: Optional[Dict] = None, 
                 channel_display_groups: Optional[Union[
                    Dict[str, Sequence[NonNegInt]], 
                    Sequence[Sequence[NonNegInt]]]] = None):
        # TODO add docs from Config classes
        # TODO validate input using functionality in Config classes
        # TODO set defaults 
        self.class_names = class_names
        self.class_colors = class_colors
        self.transform = transform
        self.channel_display_groups = channel_display_groups

    @abstractmethod
    def plot_xyz(self, axs, x: Tensor, y, z=None):
        """Plot image, ground truth labels, and predicted labels.

        Args:
            axs: matplotlib axes on which to plot
            x: image
            y: ground truth labels
            z: optional predicted labels
        """
        pass

    def get_plot_nrows(self, **kwargs) -> int:
        x = kwargs['x']
        batch_limit = kwargs.get('batch_limit')
        batch_sz, c, h, w = x.shape
        nrows = min(batch_sz,
                    batch_limit) if batch_limit is not None else batch_sz
        return nrows

    def get_plot_ncols(self, **kwargs) -> int:
        ncols = len(self.channel_display_groups)
        return ncols

    def get_plot_params(self, **kwargs) -> dict:
        nrows = self.get_plot_nrows(**kwargs)
        ncols = self.get_plot_ncols(**kwargs)
        params = {
            'fig_args': {
                'nrows': nrows,
                'ncols': ncols,
                'constrained_layout': True,
                'figsize': (3 * ncols, 3 * nrows),
                'squeeze': False
            },
            'plot_xyz_args': {}
        }
        return params

    def plot_batch(self,
                   x: Tensor,
                   y: Sequence,
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
        if params['fig_args']['nrows'] == 0:
            return

        fig, axs = plt.subplots(**params['fig_args'])

        # (N, c, h, w) --> (N, h, w, c)
        x = x.permute(0, 2, 3, 1)

        # apply transform, if given
        if self.transform is not None:
            tf = deserialize_albumentation_transform(self.transform)
            imgs = [tf(image=img)['image'] for img in x.numpy()]
            x = torch.from_numpy(np.stack(imgs))

        plot_xyz_args = params['plot_xyz_args']
        for i, row_axs in enumerate(axs):
            if z is None:
                self.plot_xyz(row_axs, x[i], y[i], **plot_xyz_args)
            else:
                self.plot_xyz(row_axs, x[i], y[i], z=z[i], **plot_xyz_args)

        if show:
            plt.show()
        if output_path is not None:
            make_dir(output_path, use_dirname=True)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)

        plt.close(fig)