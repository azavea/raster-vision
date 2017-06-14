import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from rastervision.common.utils import (
    expand_dims, compute_ndvi, plot_img_row, download_dataset)
from rastervision.common.data.generators import Batch, FileGenerator

ISPRS = 'isprs'


class IsprsBatch(Batch):
    def __init__(self):
        self.y_mask = None

        super().__init__()


class IsprsDataset():
    """Metadata and utilities for dealing with ISPRS data.

    The ISPRS semantic labeling datasets can be found at
    http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html
    The ground truth label images can be represented in several ways: 1) the
    contest organizers provide the ground truth as RGB images, where each RGB
    value represents a different label. 2) When evaluating the output of the
    model, it is more convenient to represent each label as an integer. 3) The
    neural network generates output that is one-hot coded. It is useful to be
    able to translate between the representations, so this contains methods to
    do so. Each method can take a batch with shape [batch_size, nb_rows,
    nb_cols, nb_channels], or a single image with shape [nb_rows, nb_cols,
    nb_channels], and the returned array will have the  same shape as the
    input.
    """
    def __init__(self):
        # RGB vectors corresponding to different labels
        # Impervious surfaces (RGB: 255, 255, 255)
        # Building (RGB: 0, 0, 255)
        # Low vegetation (RGB: 0, 255, 255)
        # Tree (RGB: 0, 255, 0)
        # Car (RGB: 255, 255, 0)
        # Clutter/background (RGB: 255, 0, 0)
        self.label_keys = [
            [255, 255, 255],
            [0, 0, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [255, 0, 0],
        ]

        self.nb_labels = len(self.label_keys)

        self.label_names = [
            'Impervious',
            'Building',
            'Low vegetation',
            'Tree',
            'Car',
            'Clutter'
        ]

    @expand_dims
    def rgb_to_mask_batch(self, batch):
        """Convert a label image with black boundary pixels into a mask.

        Since there is uncertainty associated with the boundary of
        objects/regions in the ground truth segmentation, it makes sense
        to ignore these boundaries during evaluation. To help, the contest
        organizers have provided special ground truth images where the boundary
        pixels (in a 3 pixel radius) are black.

        # Returns
            A boolean array where an element is True if it should be used in
            the evaluation, and ignored otherwise.
        """
        mask = (batch[:, :, :, 0] == 0) & \
               (batch[:, :, :, 1] == 0) & \
               (batch[:, :, :, 2] == 0)
        mask = np.bitwise_not(mask)
        mask = np.expand_dims(mask, axis=3)

        return mask

    @expand_dims
    def rgb_to_label_batch(self, batch):
        label_batch = np.zeros(batch.shape[:-1])
        for label, key in enumerate(self.label_keys):
            mask = (batch[:, :, :, 0] == key[0]) & \
                   (batch[:, :, :, 1] == key[1]) & \
                   (batch[:, :, :, 2] == key[2])
            label_batch[mask] = label

        return np.expand_dims(label_batch, axis=3)

    @expand_dims
    def label_to_one_hot_batch(self, label_batch):
        if label_batch.ndim == 4:
            label_batch = np.squeeze(label_batch, axis=3)

        nb_labels = len(self.label_keys)
        shape = np.concatenate([label_batch.shape, [nb_labels]])
        one_hot_batch = np.zeros(shape)

        for label in range(nb_labels):
            one_hot_batch[:, :, :, label][label_batch == label] = 1.
        return one_hot_batch

    @expand_dims
    def rgb_to_one_hot_batch(self, rgb_batch):
        label_batch = self.rgb_to_label_batch(rgb_batch)
        return self.label_to_one_hot_batch(label_batch)

    @expand_dims
    def label_to_rgb_batch(self, label_batch):
        if label_batch.ndim == 4:
            label_batch = np.squeeze(label_batch, axis=3)

        rgb_batch = np.zeros(np.concatenate([label_batch.shape, [3]]),
                             dtype=np.uint8)
        for label, key in enumerate(self.label_keys):
            mask = label_batch == label
            rgb_batch[mask, :] = key

        return rgb_batch

    @expand_dims
    def one_hot_to_label_batch(self, one_hot_batch):
        one_hot_batch = np.argmax(one_hot_batch, axis=3)
        return np.expand_dims(one_hot_batch, axis=3)

    @expand_dims
    def one_hot_to_rgb_batch(self, one_hot_batch):
        label_batch = self.one_hot_to_label_batch(one_hot_batch)
        return self.label_to_rgb_batch(label_batch)

    def augment_channels(self, batch_x):
        red = batch_x[:, :, :, [self.red_ind]]
        ir = batch_x[:, :, :, [self.ir_ind]]
        ndvi = compute_ndvi(red, ir)
        return np.concatenate([batch_x, ndvi], axis=3)


class IsprsFileGenerator(FileGenerator):
    def __init__(self, options):
        super().__init__(options)

    def plot_sample(self, file_path, x, y):
        fig = plt.figure()
        nb_cols = max(self.dataset.nb_channels + 1, self.dataset.nb_labels + 1)
        grid_spec = mpl.gridspec.GridSpec(2, nb_cols)

        # Plot x channels
        x = self.calibrate_image(x)
        rgb_x = x[:, :, self.dataset.rgb_inds]
        imgs = [rgb_x]
        nb_channels = x.shape[2]
        for channel_ind in range(nb_channels):
            img = x[:, :, channel_ind]
            imgs.append(img)
        row_ind = 0
        plot_img_row(fig, grid_spec, row_ind, imgs)

        # Plot y channels
        rgb_y = self.dataset.one_hot_to_rgb_batch(y)
        imgs = [rgb_y]
        for channel_ind in range(y.shape[2]):
            img = y[:, :, channel_ind]
            imgs.append(img)
        row_ind = 1
        plot_img_row(fig, grid_spec, row_ind, imgs)

        plt.savefig(file_path, bbox_inches='tight', format='pdf', dpi=600)
        plt.close(fig)

    def download_dataset(self, file_names):
        download_dataset(ISPRS, file_names)
