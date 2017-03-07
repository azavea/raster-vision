from os import makedirs

# For some reason, you need to import PIL first.
from PIL import Image
import numpy as np
import rasterio

from .settings import label_keys, nb_labels


def _makedirs(path):
    try:
        makedirs(path)
    except:
        pass


def load_tiff(file_path):
    with rasterio.open(file_path, 'r+') as r:
        return np.transpose(r.read(), axes=[1, 2, 0])


def load_image(file_path):
    im = Image.open(file_path)
    return np.array(im)


def save_image(file_path, im):
    np.save(file_path, im.astype(np.uint8))


def rgb_to_label_batch(rgb_batch):
    label_batch = np.zeros(rgb_batch.shape[:-1])
    for label, key in enumerate(label_keys):
        mask = (rgb_batch[:, :, :, 0] == key[0]) & \
               (rgb_batch[:, :, :, 1] == key[1]) & \
               (rgb_batch[:, :, :, 2] == key[2])
        label_batch[mask] = label

    return label_batch


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


def label_to_one_hot_batch(label_batch):
    one_hot_batch = np.zeros(np.concatenate([label_batch.shape, [nb_labels]]))
    for label in range(nb_labels):
        one_hot_batch[:, :, :, label][label_batch == label] = 1.
    return one_hot_batch

def rgb_to_one_hot_batch(rgb_batch):
    return label_to_one_hot_batch(rgb_to_label_batch(rgb_batch))


def label_to_rgb_batch(label_batch):
    rgb_batch = np.zeros(np.concatenate([label_batch.shape, [3]]))
    for label, key in enumerate(label_keys):
        mask = label_batch == label
        rgb_batch[mask, :] = key

    return rgb_batch


def one_hot_to_label_batch(one_hot_batch):
    return np.argmax(one_hot_batch, axis=3)


def one_hot_to_rgb_batch(one_hot_batch):
    return label_to_rgb_batch(one_hot_to_label_batch(one_hot_batch))
