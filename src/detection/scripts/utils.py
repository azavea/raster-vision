import numpy as np


def load_window(image_dataset, window=None):
    im = np.transpose(
        image_dataset.read(window=window), axes=[1, 2, 0])
    # XXX this is specific to Planet Labs imagery which is bgr-ir
    im = im[:, :, [2, 1, 0]]
    return im
