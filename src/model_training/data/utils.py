from os import makedirs

# For some reason, you need to import PIL first.
from PIL import Image
import numpy as np
import rasterio
from rasterio.profiles import default_gtiff_profile


def _makedirs(path):
    try:
        makedirs(path)
    except:
        pass


def load_tiff(file_path):
    with rasterio.open(file_path, 'r+') as r:
        return np.transpose(r.read(), axes=[1, 2, 0])


def save_tiff(im, file_path):
    height, width, count = im.shape
    with rasterio.open(file_path, 'w', driver='GTiff', height=height,
                       width=width, count=3, dtype=np.uint8) as dst:
        dst.write(im[:, :, 0], 1)
        dst.write(im[:, :, 1], 2)
        dst.write(im[:, :, 2], 3)


def load_image(file_path):
    im = Image.open(file_path)
    return np.array(im)


def save_image(file_path, im):
    np.save(file_path, im.astype(np.uint8))


def rgb_to_mask(im):
    """
    Used to convert a label image where boundary pixels are black into
    an image where non-boundary pixel are true and boundary pixels are false.
    """
    mask = (im[:, :, 0] == 0) & \
           (im[:, :, 1] == 0) & \
           (im[:, :, 2] == 0)
    mask = np.bitwise_not(mask)

    return mask


def rgb_to_label_batch(rgb_batch, label_keys):
    label_batch = np.zeros(rgb_batch.shape[:-1])
    for label, key in enumerate(label_keys):
        mask = (rgb_batch[:, :, :, 0] == key[0]) & \
               (rgb_batch[:, :, :, 1] == key[1]) & \
               (rgb_batch[:, :, :, 2] == key[2])
        label_batch[mask] = label

    return label_batch


def label_to_one_hot_batch(label_batch, label_keys):
    nb_labels = len(label_keys)
    one_hot_batch = np.zeros(np.concatenate([label_batch.shape, [nb_labels]]))
    for label in range(nb_labels):
        one_hot_batch[:, :, :, label][label_batch == label] = 1.
    return one_hot_batch


def rgb_to_one_hot_batch(rgb_batch, label_keys):
    label_batch = rgb_to_label_batch(rgb_batch, label_keys)
    return label_to_one_hot_batch(label_batch, label_keys)


def label_to_rgb_batch(label_batch, label_keys):
    rgb_batch = np.zeros(np.concatenate([label_batch.shape, [3]]))
    for label, key in enumerate(label_keys):
        mask = label_batch == label
        rgb_batch[mask, :] = key

    return rgb_batch


def one_hot_to_label_batch(one_hot_batch):
    return np.argmax(one_hot_batch, axis=3)


def one_hot_to_rgb_batch(one_hot_batch, label_keys):
    label_batch = one_hot_to_label_batch(one_hot_batch)
    return label_to_rgb_batch(label_batch, label_keys)


def safe_divide(a, b):
    """
    Avoid divide by zero
    http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[c == np.inf] = 0
        c = np.nan_to_num(c)
        return c


def compute_ndvi(red, ir):
    ndvi = safe_divide((ir - red), (ir + red))
    ndvi = np.expand_dims(ndvi, axis=3)
    return ndvi
