from os.path import join
import json

import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics

from ..data.utils import _makedirs, safe_divide, predict_img
from ..data.generators import VALIDATION
from .utils import make_prediction_img


VALIDATION_EVAL = 'validation_eval'


class Scores():
    """A set of scores for the performance of a model on a dataset."""

    def __init__(self):
        pass

    def compute_scores(self, label_names, confusion_mat):
        """Compute scores from a confusion matrix.

        # Arguments
            label_names: a list of labels (ie. class) names
            confusion_mat: a 2D array of size [nb_labels, nb_labels]
                containing a confusion matrix
        """
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


def compute_confusion_mat(ground_truth, ground_truth_mask, predictions,
                          nb_labels):
    """Compute a confusion matrix for an image.

    # Arguments
        ground_truth: ground truth label array of size [nb_rows, nb_cols]
        ground_truth_mask: boolean array of size [nb_rows, nb_cols] that is
            true for pixels that should be used in the evaluation.
        predictions: a label array of size [nb_rows, nb_cols] that has the
            values predicted by the model
        nb_labels: the number of labels in the dataset

    # Returns
        A confusion matrix of size [nb_labels, nb_labels]
    """
    ground_truth = np.ravel(ground_truth)
    predictions = np.ravel(predictions)
    ground_truth_mask = np.ravel(ground_truth_mask).astype(np.bool)

    ground_truth = ground_truth[ground_truth_mask]
    predictions = predictions[ground_truth_mask]

    return metrics.confusion_matrix(
        ground_truth, predictions, labels=np.arange(nb_labels))


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


def plot_prediction(dataset, predictions_path, sample_index, display_batch_x,
                    display_batch_y, display_predictions, is_debug=False):
    fig = plt.figure()

    nb_subplot_cols = 3
    if is_debug:
        nb_subplot_cols += dataset.nb_channels

    gs = mpl.gridspec.GridSpec(1, nb_subplot_cols)

    def plot_img(subplot_index, im, title, is_rgb=False):
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
    rgb_input_im = display_batch_x[:, :, dataset.rgb_input_inds]
    plot_img(subplot_index, rgb_input_im, 'RGB', is_rgb=True)

    if is_debug:
        if dataset.include_ir:
            subplot_index += 1
            ir_im = display_batch_x[:, :, dataset.ir_ind]
            plot_img(subplot_index, ir_im, 'IR')

        if dataset.include_depth:
            subplot_index += 1
            depth_im = display_batch_x[:, :, dataset.depth_ind]
            plot_img(subplot_index, depth_im, 'Depth')

        if dataset.include_ndvi:
            subplot_index += 1
            ndvi_im = display_batch_x[:, :, dataset.ndvi_ind]
            ndvi_im = (np.clip(ndvi_im, -1, 1) + 1) * 100
            plot_img(subplot_index, ndvi_im, 'NDVI')

    subplot_index += 1
    plot_img(subplot_index, display_batch_y[:, :, :], 'Ground Truth',
               is_rgb=True)
    subplot_index += 1
    plot_img(subplot_index, display_predictions[:, :, :], 'Prediction',
               is_rgb=True)

    make_legend(dataset.label_keys, dataset.label_names)

    file_name = '{}_debug.pdf' if is_debug else '{}.pdf'
    file_name = file_name.format(sample_index)
    file_path = join(predictions_path, file_name)
    plt.savefig(file_path, bbox_inches='tight', format='pdf', dpi=300)

    plt.close(fig)


def save_scores(scores, run_path):
    with open(join(run_path, 'scores.txt'), 'w') as scores_file:
        scores_file.write(scores.to_json())


def validation_eval(run_path, model, options, generator):
    """Evaluate model on validation data.

    For each validation image, make a prediction, plot the prediction along
    with the ground truth, and increment a confusion matrix. After all
    validation images have been processed, compute and save scores based on the
    confusion matrix. This allows us to compute scores for datasets that cannot
    fit into memory.

    # Arguments
        run_path: the path to the files for a run
        model: a Keras model that has been trained
        options: RunOptions object that specifies the run
        generator: a Generator object to generate the test data
    """
    dataset = generator.dataset
    label_names = dataset.label_names

    validation_gen = generator.make_split_generator(
        VALIDATION, target_size=options.eval_target_size,
        batch_size=1, shuffle=False, augment=False, normalize=True,
        eval_mode=True)

    confusion_mat = np.zeros((dataset.nb_labels, dataset.nb_labels))
    predictions_path = join(run_path, 'validation_predictions')
    _makedirs(predictions_path)

    for sample_index, (batch_x, batch_y, batch_y_mask, file_ind) in \
            enumerate(validation_gen):
        file_ind = file_ind[0]
        print('Processing {}'.format(file_ind))

        batch_x = np.squeeze(batch_x, axis=0)
        batch_y = np.squeeze(batch_y, axis=0)
        batch_y_mask = np.squeeze(batch_y_mask, axis=0)

        display_batch_x = generator.unnormalize(batch_x)
        display_batch_y = dataset.one_hot_to_rgb_batch(batch_y)
        display_predictions = make_prediction_img(
            batch_x, options.target_size[0],
            lambda x: dataset.one_hot_to_rgb_batch(predict_img(x, model)))

        label_batch_y = dataset.one_hot_to_label_batch(batch_y)
        label_predictions = dataset.rgb_to_label_batch(display_predictions)

        plot_prediction(
            dataset, predictions_path, sample_index, display_batch_x,
            display_batch_y, display_predictions)
        plot_prediction(
            dataset, predictions_path, sample_index, display_batch_x,
            display_batch_y, display_predictions, is_debug=True)

        confusion_mat += compute_confusion_mat(
            label_batch_y, batch_y_mask, label_predictions, dataset.nb_labels)

        if (options.nb_eval_samples is not None and
                sample_index == options.nb_eval_samples - 1):
            break

    scores = Scores()
    scores.compute_scores(label_names, confusion_mat)
    save_scores(scores, run_path)
