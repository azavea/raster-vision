"""Utility functions shared across tasks."""
import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rastervision.common.utils import plot_img_row


def predict_x(x, model):
    batch_x = np.expand_dims(x, axis=0)
    batch_y = model.predict(batch_x)
    y = np.squeeze(batch_y, axis=0)
    return y


def make_prediction_img(x, target_size, predict):
    """Generate a prediction image one window at a time.

    Generate a prediction image consisting of a prediction for each pixel. The
    format of that prediction depends on the output of the predict function.
    Passing a very large image as input to a model might
    not be possible due to memory limitations. Instead, we slide a window over
    the image and get the predictions for each window. The individual
    predictions can be combined to create a large prediction image. By
    overlapping the windows, we can discard inaccurate predictions along window
    boundaries.

    # Arguments
        x: the full sized image to get a prediction for
            (nb_rows, nb_cols, nb_channels)
        target_size: the window size which needs to be the same as what the
            model expects as input
        predict: a function that takes a window image of size
            target_size and returns the prediction for each pixel

    # Returns
        The prediction image
    """
    quarter_target_size = target_size // 4
    half_target_size = target_size // 2
    sample_prediction = predict(x[0:target_size, 0:target_size, :])
    nb_channels = sample_prediction.shape[2]
    dtype = sample_prediction.dtype

    pad_width = (
        (quarter_target_size, target_size),
        (quarter_target_size, target_size),
        (0, 0))

    pad_x = np.pad(x, pad_width, 'edge')
    pad_y = np.zeros(
        (pad_x.shape[0], pad_x.shape[1], nb_channels),
        dtype=dtype)

    def update_prediction_center(row_begin, row_end, col_begin, col_end):
        """Just update the center half of the window."""

        x_window = pad_x[row_begin:row_end, col_begin:col_end, :]
        y_window = predict(x_window)

        y_window_center = y_window[
            quarter_target_size:target_size - quarter_target_size,
            quarter_target_size:target_size - quarter_target_size,
            :]

        pad_y[
           row_begin + quarter_target_size:row_end - quarter_target_size,
           col_begin + quarter_target_size:col_end - quarter_target_size,
           :] = y_window_center

    for row_begin in range(0, pad_x.shape[0], half_target_size):
        for col_begin in range(0, pad_x.shape[1], half_target_size):
            row_end = row_begin + target_size
            col_end = col_begin + target_size
            if row_end <= pad_x.shape[0] and col_end <= pad_x.shape[1]:
                update_prediction_center(
                    row_begin, row_end, col_begin, col_end)

    y = pad_y[quarter_target_size:quarter_target_size+x.shape[0],
              quarter_target_size:quarter_target_size+x.shape[1],
              :]
    return y


def make_legend(label_keys, label_names):
    patches = []
    for label_key, label_name in zip(label_keys, label_names):
        color = tuple(np.array(label_key) / 255.)
        patch = mpatches.Patch(
            facecolor=color, edgecolor='black', linewidth=0.5,
            label=label_name)
        patches.append(patch)
    plt.legend(handles=patches, loc='upper left',
               bbox_to_anchor=(1, 1), fontsize=4)


def plot_prediction(generator, all_x, y, pred,
                    file_path, is_debug=False):
    dataset = generator.dataset
    fig = plt.figure()

    nb_subplot_cols = 3
    if is_debug:
        nb_subplot_cols += len(generator.active_input_inds)

    grid_spec = mpl.gridspec.GridSpec(1, nb_subplot_cols)

    all_x = generator.calibrate_image(all_x)
    rgb_input_im = all_x[:, :, dataset.rgb_inds]
    imgs = [rgb_input_im]
    titles = ['RGB']

    if is_debug:
        ir_im = all_x[:, :, dataset.ir_ind]
        imgs.append(ir_im)
        titles.append('IR')

        depth_im = all_x[:, :, dataset.depth_ind]
        imgs.append(depth_im)
        titles.append('Depth')

        ndvi_im = all_x[:, :, dataset.ndvi_ind]
        imgs.append(ndvi_im)
        titles.append('NDVI')

    imgs.append(y)
    titles.append('Ground Truth')

    imgs.append(pred)
    titles.append('Prediction')

    plot_img_row(fig, grid_spec, 0, imgs, titles)
    make_legend(dataset.label_keys, dataset.label_names)
    plt.savefig(file_path, bbox_inches='tight', format='png', dpi=300)

    plt.close(fig)
