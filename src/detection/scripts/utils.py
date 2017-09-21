import numpy as np


def load_window(image_dataset, channel_order, window=None):
    im = np.transpose(
        image_dataset.read(window=window), axes=[1, 2, 0])
    im = im[:, :, channel_order]
    return im
