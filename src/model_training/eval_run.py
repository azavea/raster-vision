"""
Compute prediction images, learning curve graph and various metrics for a run.
"""
from os.path import join
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
    _makedirs)
from .data.generators import (
    make_split_generator, unscale_inputs, load_channel_stats)
from .data.settings import (
    label_names, label_keys, results_path, VALIDATION, get_dataset_info)


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
        self.precision = true_pos / (true_pos + false_pos)
        self.recall = true_pos / (true_pos + false_neg)
        self.f1 = 2 * ((self.precision * self.recall) /
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


def make_legend():
    patches = []
    for label_key, label_name in zip(label_keys, label_names):
        color = tuple(np.array(label_key) / 255.)
        patch = mpatches.Patch(
            facecolor=color, edgecolor='black', linewidth=0.5,
            label=label_name)
        patches.append(patch)
    plt.legend(handles=patches, loc='upper left',
               bbox_to_anchor=(1, 1), fontsize=4)


def plot_prediction(run_path, sample_index, display_inputs, display_outputs,
                    display_predictions):
    fig = plt.figure()
    nb_subplot_cols = 3
    gs = mpl.gridspec.GridSpec(1, nb_subplot_cols)

    def plot_image(subplot_index, im, title):
        a = fig.add_subplot(gs[subplot_index])
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)

        a.imshow(im.astype(np.uint8))
        if subplot_index < nb_subplot_cols:
            a.set_title(title, fontsize=6)

    subplot_index = 0
    plot_image(subplot_index, display_inputs[0, :, :, :], 'Input')
    subplot_index = 1
    plot_image(subplot_index, display_outputs[0, :, :, :], 'Ground Truth')
    subplot_index = 2
    plot_image(subplot_index, display_predictions[0, :, :, :], 'Prediction')

    make_legend()
    predictions_path = join(
        run_path, 'predictions', '{}.pdf'.format(sample_index))
    plt.savefig(predictions_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.close(fig)


def compute_predictions(model, run_path, options, dataset_info):
    _makedirs(join(run_path, 'predictions'))

    tile_size = dataset_info.input_shape[0:2]
    validation_gen = make_split_generator(
        dataset_info, VALIDATION, batch_size=1, shuffle=False, augment=False,
        scale=True, eval_mode=True)
    scale_params = load_channel_stats(dataset_info.dataset_path)

    confusion_mat = np.zeros((dataset_info.nb_labels, dataset_info.nb_labels))

    for sample_index, (inputs, outputs, outputs_mask) in enumerate(validation_gen):
        print('.')
        predictions = model.predict(inputs)

        display_inputs = unscale_inputs(inputs, dataset_info.input_inds, scale_params)
        display_inputs = display_inputs[:, :, :, dataset_info.rgb_input_inds]
        display_outputs = label_to_rgb_batch(np.squeeze(outputs, axis=3))
        display_predictions = one_hot_to_rgb_batch(predictions)

        plot_prediction(
            run_path, sample_index, display_inputs, display_outputs,
            display_predictions)

        confusion_mat += compute_confusion_mat(
            outputs, one_hot_to_label_batch(predictions), outputs_mask,
            dataset_info.nb_labels)

    scores = Scores()
    scores.compute_scores(label_names, confusion_mat)
    save_scores(scores, run_path)


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

    print('Generating predictions and scores...')
    compute_predictions(
        model, run_path, options, dataset_info)

    print('Plotting graphs...')
    plot_graphs(model, run_path)
