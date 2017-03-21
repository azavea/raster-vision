from os import makedirs
from os.path import splitext

# For some reason, you need to import PIL first.
from PIL import Image
import numpy as np
import rasterio
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt


def _makedirs(path):
    try:
        makedirs(path)
    except:
        pass


def load_rasterio(file_path, window=None):
    with rasterio.open(file_path, 'r+') as r:
        return np.transpose(r.read(window=window), axes=[1, 2, 0])


def load_pillow(file_path, window=None):
    im = Image.open(file_path)
    if window is not None:
        ((row_begin, row_end), (col_begin, col_end)) = window
        box = (col_begin, row_begin, col_end, row_end)
        im = im.crop(box)
    im = np.array(im)
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    return im


def load_image(file_path, window=None):
    ext = splitext(file_path)[1]
    if ext in ['.tif', '.tiff']:
        return load_rasterio(file_path, window)
    return load_pillow(file_path, window)


def get_rasterio_size(file_path):
    with rasterio.open(file_path, 'r+') as r:
        nb_rows, nb_cols = r.height, r.width
        return nb_rows, nb_cols


def get_pillow_size(file_path):
    im = Image.open(file_path)
    nb_cols, nb_rows = im.size
    return nb_cols, nb_rows


def get_image_size(file_path):
    ext = splitext(file_path)[1]
    if ext in ['.tif', '.tiff']:
        return get_rasterio_size(file_path)
    return get_pillow_size(file_path)


def save_rasterio(im, file_path):
    # TODO turn compression off
    height, width, count = im.shape
    with rasterio.open(file_path, 'w', driver='GTiff', height=height,
                       width=width, count=3, dtype=np.uint8) as dst:
        dst.write(im[:, :, 0], 1)
        dst.write(im[:, :, 1], 2)
        dst.write(im[:, :, 2], 3)


def save_pillow(im, file_path):
    im = Image.fromarray(im)
    im.save(file_path)


def save_image(im, file_path):
    ext = splitext(file_path)[1]
    if ext in ['.tif', '.tiff']:
        save_rasterio(im, file_path)
    else:
        save_pillow(im, file_path)


def save_numpy_array(file_path, arr):
    np.save(file_path, arr.astype(np.uint8))


def expand_dims(func):
    def wrapper(self, batch):
        ndim = batch.ndim
        if ndim == 3:
            batch = np.expand_dims(batch, axis=0)
        batch = func(self, batch)
        if ndim == 3:
            batch = np.squeeze(batch, axis=0)
        return batch
    return wrapper


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
    return ndvi


def get_channel_stats(batch):
    nb_channels = batch.shape[3]
    channel_data = np.reshape(
        np.transpose(batch, [3, 0, 1, 2]), (nb_channels, -1))

    means = np.mean(channel_data, axis=1)
    stds = np.std(channel_data, axis=1)
    return (means, stds)


def predict_image(image, model):
    batch = np.expand_dims(image, axis=0)
    outputs = model.predict(batch)
    outputs = np.squeeze(outputs, axis=0)
    return outputs


def plot_sample(file_path, inputs, outputs, generator):
    inputs = generator.unnormalize_inputs(inputs)
    dataset = generator.dataset

    fig = plt.figure()
    nb_input_inds = inputs.shape[2]
    nb_output_inds = outputs.shape[2]

    gs = mpl.gridspec.GridSpec(2, 7)

    def plot_image(plot_row, plot_col, im, is_rgb=False):
        a = fig.add_subplot(gs[plot_row, plot_col])
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)

        if is_rgb:
            a.imshow(im.astype(np.uint8))
        else:
            a.imshow(im, cmap='gray', vmin=0, vmax=255)

    plot_row = 0
    plot_col = 0
    im = inputs[:, :, dataset.rgb_input_inds]
    plot_image(plot_row, plot_col, im, is_rgb=True)

    for channel_ind in range(nb_input_inds):
        plot_col += 1
        if channel_ind == dataset.ndvi_ind:
            im = (np.clip(inputs[:, :, channel_ind], -1, 1) + 1) * 100
        else:
            im = inputs[:, :, channel_ind]
        plot_image(plot_row, plot_col, im)

    plot_row = 1
    plot_col = 0
    rgb_outputs = dataset.one_hot_to_rgb_batch(outputs)
    plot_image(plot_row, plot_col, rgb_outputs, is_rgb=True)

    for channel_ind in range(nb_output_inds):
        plot_col += 1
        im = outputs[:, :, channel_ind] * 150
        plot_image(plot_row, plot_col, im)

    plt.savefig(file_path, bbox_inches='tight', format='pdf', dpi=600)
    plt.close(fig)
