import rasterio
import numpy as np


def load_tiff(image_path, window=None):
    image_dataset = rasterio.open(image_path)
    im = np.transpose(
        image_dataset.read(window=window), axes=[1, 2, 0])
    # XXX is this specific to the dataset?
    # bgr-ir
    im = im[:, :, [2, 1, 0]]
    return im, image_dataset
