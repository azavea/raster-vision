import numpy as np

from .utils import expand_dims


class IsprsDataset():
    def __init__(self):
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
            'Unknown'
        ]

    @expand_dims
    def rgb_to_mask_batch(self, batch):
        """
        Used to convert a label image where boundary pixels are black into
        an image where non-boundary pixel are true and boundary pixels are
        false.
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

        rgb_batch = np.zeros(np.concatenate([label_batch.shape, [3]]))
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
