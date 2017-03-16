"""
Compute prediction images, learning curve graph and various metrics for a run.
"""
from os.path import join, splitext
import json

import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics

from .data.utils import (
    one_hot_to_rgb_batch, one_hot_to_label_batch, label_to_rgb_batch,
    rgb_to_label_batch, _makedirs, safe_divide, save_tiff)
from .data.generators import (
    make_split_generator, unscale_inputs, load_channel_stats,
    make_test_generator)
from .data.settings import (results_path, VALIDATION)


class Scores():
    def __init__(self):
        pass

    def compute_scores(self, label_names, confusion_mat):
        self.label_names = label_names
        self.confusion_mat = confusion_mat

        true_pos = np.diagonal(self.confusion_mat)
        false_pos = np.sum(self.confusion_mat, axis=0) - true_pos
        false_neg = np.sum(self.confusion_mat, axis=1) - true_pos
        self.support = np.sum(self.confusion_mat, axis=1)
        self.precision = safe_divide(true_pos, (true_pos + false_pos))
        self.recall = safe_divide(true_pos, (true_pos + false_neg))
        self.f1 = 2 * safe_divide((self.precision * self.recall),
                                  (self.precision + self.recall))
        self.avg_accuracy = np.sum(true_pos) / np.sum(self.support)

    def to_json(self):
        scores = Scores()
        scores.label_names = self.label_names
        scores.confusion_mat = self.confusion_mat.tolist()
        scores.support = self.support.tolist()
        scores.precision = self.precision.tolist()
        scores.recall = self.recall.tolist()
        scores.f1 = self.f1.tolist()
        scores.avg_accuracy = self.avg_accuracy

        return json.dumps(scores.__dict__, sort_keys=True, indent=4)


def compute_confusion_mat(outputs, predictions, outputs_mask, nb_labels):
    outputs = np.ravel(outputs)
    predictions = np.ravel(predictions)
    outputs_mask = np.ravel(outputs_mask).astype(np.bool)

    outputs = outputs[outputs_mask]
    predictions = predictions[outputs_mask]

    return metrics.confusion_matrix(
        outputs, predictions, labels=np.arange(nb_labels))


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


def plot_prediction(dataset_info, predictions_path, sample_index, display_inputs,
                    display_outputs, display_predictions, is_debug=False):
    fig = plt.figure()

    nb_subplot_cols = 3
    if is_debug:
        nb_subplot_cols += dataset_info.include_ir + \
            dataset_info.include_depth + dataset_info.include_ndvi

    gs = mpl.gridspec.GridSpec(1, nb_subplot_cols)

    def plot_image(subplot_index, im, title, is_rgb=False):
        a = fig.add_subplot(gs[subplot_index])
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)

        if is_rgb:
            a.imshow(im.astype(np.uint8))
        else:
            a.imshow(im, cmap='gray', vmin=0, vmax=255)
        if subplot_index < nb_subplot_cols:
            a.set_title(title, fontsize=6)

    subplot_index = 0
    display_inputs = display_inputs[0, :, :, :]
    rgb_input_im = display_inputs[:, :, dataset_info.rgb_input_inds]
    plot_image(subplot_index, rgb_input_im, 'RGB', is_rgb=True)

    if is_debug:
        if dataset_info.include_ir:
            subplot_index += 1
            ir_im = display_inputs[:, :, dataset_info.ir_ind]
            plot_image(subplot_index, ir_im, 'IR')

        if dataset_info.include_depth:
            subplot_index += 1
            depth_im = display_inputs[:, :, dataset_info.depth_ind]
            plot_image(subplot_index, depth_im, 'Depth')

        if dataset_info.include_ndvi:
            subplot_index += 1
            nvdi_im = display_inputs[:, :, dataset_info.ndvi_ind]
            plot_image(subplot_index, nvdi_im, 'NDVI')

    subplot_index += 1
    plot_image(subplot_index, display_outputs[0, :, :, :], 'Ground Truth',
               is_rgb=True)
    subplot_index += 1
    plot_image(subplot_index, display_predictions[0, :, :, :], 'Prediction',
               is_rgb=True)

    make_legend(dataset_info.label_keys, dataset_info.label_names)

    file_name = '{}_debug.pdf' if is_debug else '{}.pdf'
    file_name = file_name.format(sample_index)
    file_path = join(predictions_path, file_name)
    plt.savefig(file_path, bbox_inches='tight', format='pdf', dpi=300)

    plt.close(fig)


def validation_eval(model, run_path, options, dataset_info):
    eval_tile_size = dataset_info.eval_tile_size
    tile_size = dataset_info.tile_size
    label_keys = dataset_info.label_keys
    label_names = dataset_info.label_names

    validation_gen = make_split_generator(
        dataset_info, VALIDATION,
        tile_size=(eval_tile_size, eval_tile_size),
        batch_size=1, shuffle=False, augment=False, scale=True, eval_mode=True)
    scale_params = load_channel_stats(dataset_info.dataset_path)

    confusion_mat = np.zeros((dataset_info.nb_labels, dataset_info.nb_labels))

    predictions_path = join(run_path, 'validation_predictions')
    _makedirs(predictions_path)

    for sample_index, (inputs, outputs, outputs_mask) in \
            enumerate(validation_gen):
        print('.')

        display_inputs = unscale_inputs(inputs, dataset_info.input_inds,
                                        scale_params)
        display_outputs = label_to_rgb_batch(
            np.squeeze(outputs, axis=3), label_keys)
        display_predictions = make_prediction_tile(
            inputs, eval_tile_size, tile_size, label_keys, model)
        display_predictions = np.expand_dims(display_predictions, axis=0)
        label_predictions = rgb_to_label_batch(
            display_predictions, label_keys)

        plot_prediction(
            dataset_info, predictions_path, sample_index, display_inputs,
            display_outputs, display_predictions)
        plot_prediction(
            dataset_info, predictions_path, sample_index, display_inputs,
            display_outputs, display_predictions, is_debug=True)

        confusion_mat += compute_confusion_mat(
            outputs, label_predictions, outputs_mask, dataset_info.nb_labels)

        if (options.nb_eval_samples is not None and
                sample_index == options.nb_eval_samples - 1):
            break

    scores = Scores()
    scores.compute_scores(label_names, confusion_mat)
    save_scores(scores, run_path)


def make_prediction_tile(full_tile, full_tile_size, tile_size, label_keys,
                         model):
    quarter_tile_size = tile_size // 4
    half_tile_size = tile_size // 2
    full_prediction_tile = \
        np.zeros((full_tile_size, full_tile_size, 3), dtype=np.uint8)

    def snap_bounds(row_begin, row_end, col_begin, col_end):
        # If the tile straddles the edge of the full_tile, then
        # snap it to the edge.
        if row_end > full_tile_size:
            row_begin = full_tile_size - tile_size
            row_end = full_tile_size

        if col_end > full_tile_size:
            col_begin = full_tile_size - tile_size
            col_end = full_tile_size

        return row_begin, row_end, col_begin, col_end

    def update_prediction(row_begin, row_end, col_begin, col_end):
        row_begin, row_end, col_begin, col_end = \
            snap_bounds(row_begin, row_end, col_begin, col_end)

        tile = full_tile[:, row_begin:row_end, col_begin:col_end, :]
        prediction_tile = np.squeeze(
            one_hot_to_rgb_batch(model.predict(tile), label_keys))
        full_prediction_tile[row_begin:row_end, col_begin:col_end, :] = \
            prediction_tile

    def update_prediction_crop(row_begin, row_end, col_begin, col_end):
        row_begin, row_end, col_begin, col_end = \
            snap_bounds(row_begin, row_end, col_begin, col_end)

        tile = full_tile[:, row_begin:row_end, col_begin:col_end, :]
        prediction_tile = model.predict(tile)
        prediction_tile = np.squeeze(
            one_hot_to_rgb_batch(prediction_tile, label_keys))

        prediction_tile_crop = prediction_tile[
            quarter_tile_size:tile_size - quarter_tile_size,
            quarter_tile_size:tile_size - quarter_tile_size,
            :]

        full_prediction_tile[
            row_begin + quarter_tile_size:row_end - quarter_tile_size,
            col_begin + quarter_tile_size:col_end - quarter_tile_size,
            :] = prediction_tile_crop

    for row_begin in range(0, full_tile_size, half_tile_size):
        for col_begin in range(0, full_tile_size, half_tile_size):
            row_end = row_begin + tile_size
            col_end = col_begin + tile_size

            is_edge = (row_begin == 0 or row_end >= full_tile_size or
                       col_begin == 0 or col_end >= full_tile_size)

            if is_edge:
                update_prediction(row_begin, row_end, col_begin, col_end)
            else:
                update_prediction_crop(row_begin, row_end, col_begin, col_end)

    return full_prediction_tile


def test_eval(model, run_path, options, dataset_info):
    label_keys = dataset_info.label_keys
    test_predictions_path = join(run_path, 'test_predictions')
    _makedirs(test_predictions_path)

    tile_size = dataset_info.tile_size
    full_tile_size = dataset_info.full_tile_size

    test_gen = make_test_generator(dataset_info)
    for sample_ind, (full_tile, file_name) in enumerate(test_gen):
        print('Processing {}'.format(file_name))

        prediction_tile = make_prediction_tile(
            full_tile, full_tile_size, tile_size, label_keys, model)

        file_base = splitext(file_name)[0]
        prediction_file_path = join(
            test_predictions_path,
            'top_potsdam_{}_label.tif'.format(file_base))
        save_tiff(prediction_tile, prediction_file_path)

        if (options.nb_eval_samples is not None and
                sample_ind == options.nb_eval_samples - 1):
            break


def save_scores(scores, run_path):
    with open(join(run_path, 'scores.txt'), 'w') as scores_file:
        scores_file.write(scores.to_json())


def plot_graphs(model, run_path):
    log_path = join(run_path, 'log.txt')

    log = np.genfromtxt(log_path, delimiter=',', skip_header=1)
    epochs = log[:, 0]
    acc = log[:, 1]
    val_acc = log[:, 3]

    plt.figure()
    plt.title('Training Log')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.grid()
    plt.plot(epochs, acc, '-', label='Training')
    plt.plot(epochs, val_acc, '--', label='Validation')

    plt.legend(loc='best')
    accuracy_path = join(run_path, 'accuracy.pdf')
    plt.savefig(accuracy_path, format='pdf', dpi=300)


def eval_run(model, options, dataset_info):
    run_path = join(results_path, options.run_name)

    print('Generating predictions and scores for validation set...')
    validation_eval(
        model, run_path, options, dataset_info)

    print('Generating predictions for test set...')
    test_eval(
        model, run_path, options, dataset_info)

    print('Plotting graphs...')
    plot_graphs(model, run_path)
