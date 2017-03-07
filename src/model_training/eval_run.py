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

    def to_json(self):
        scores = Scores()
        scores.label_names = self.label_names
        scores.jaccard = self.jaccard
        scores.avg_f1 = self.avg_f1
        scores.avg_accuracy = self.avg_accuracy
        scores.precision = self.precision.tolist()
        scores.recall = self.recall.tolist()
        scores.f1 = self.f1.tolist()
        scores.support = self.support.tolist()
        scores.confusion_mat = self.confusion_mat.tolist()
        scores.accuracy = self.accuracy.tolist()

        return json.dumps(scores.__dict__, sort_keys=True, indent=4)


def compute_scores(outputs, predictions, outputs_mask, nb_labels):
    # Treat each pixel as a separate data point so we can use metric functions.
    outputs = np.ravel(outputs)
    predictions = np.ravel(predictions)
    outputs_mask = np.ravel(outputs_mask).astype(np.bool)

    # Remove boundary pixels
    outputs = outputs[outputs_mask]
    predictions = predictions[outputs_mask]

    # Force each image to have at least one pixel of each label so that
    # there will be an element for each label. This makes the calculation
    # slightly innaccurate but shouldn't even show up after rounding.
    # Maybe we should do something more rigorous if we have time.
    bogus_pixels = np.arange(0, nb_labels)
    outputs = np.concatenate([outputs, bogus_pixels])
    predictions = np.concatenate([predictions, bogus_pixels])

    scores = Scores()
    scores.label_names = label_names
    scores.jaccard = metrics.jaccard_similarity_score(outputs, predictions)
    scores.avg_f1 = metrics.f1_score(outputs, predictions, average='macro')
    scores.avg_accuracy = metrics.accuracy_score(outputs, predictions)
    scores.precision, scores.recall, scores.f1, scores.support = \
        metrics.precision_recall_fscore_support(outputs, predictions)
    scores.confusion_mat = metrics.confusion_matrix(outputs, predictions)
    # Avoid divide by zero error by adding 0.1
    scores.accuracy = scores.confusion_mat.diagonal() / (scores.support + 0.1)

    return scores


def get_attr_array(name, a_list):
    return np.array(list(map(lambda el: getattr(el, name), a_list)))


def aggregate_scores(scores_list):
    image_sizes = np.array(
        list(map(lambda scores: np.sum(scores.support), scores_list)))
    image_weights = image_sizes / np.sum(image_sizes)

    agg_scores = Scores()

    agg_scores.jaccard = np.sum(
        get_attr_array('jaccard', scores_list) * image_weights)
    agg_scores.avg_f1 = np.sum(
        get_attr_array('avg_f1', scores_list) * image_weights)
    agg_scores.avg_accuracy = np.sum(
        get_attr_array('avg_accuracy', scores_list) * image_weights)

    agg_scores.precision = np.sum(
        get_attr_array('precision', scores_list)
        * image_weights[:, np.newaxis],
        axis=0)
    agg_scores.recall = np.sum(
        get_attr_array('recall', scores_list)
        * image_weights[:, np.newaxis],
        axis=0)
    agg_scores.f1 = np.sum(
        get_attr_array('f1', scores_list)
        * image_weights[:, np.newaxis],
        axis=0)
    agg_scores.support = np.sum(
        get_attr_array('support', scores_list)
        * image_weights[:, np.newaxis],
        axis=0)
    agg_scores.accuracy = np.sum(
        get_attr_array('accuracy', scores_list)
        * image_weights[:, np.newaxis],
        axis=0)

    agg_scores.confusion_mat = np.sum(
        get_attr_array('confusion_mat', scores_list)
        * image_weights[:, np.newaxis, np.newaxis],
        axis=0)

    agg_scores.label_names = scores_list[0].label_names

    return agg_scores


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


def compute_predictions(model, dataset_info, run_path, options):
    _makedirs(join(run_path, 'predictions'))

    tile_size = options.input_shape[0:2]
    validation_gen = make_split_generator(
        options.dataset, VALIDATION, tile_size=tile_size,
        batch_size=1, shuffle=False, augment=False, scale=True,
        include_ir=options.include_ir, include_depth=options.include_depth,
        eval_mode=True)
    rgb_input_inds, input_inds, _, _ = dataset_info.get_channel_inds(
        include_ir=options.include_ir, include_depth=options.include_depth)
    scale_params = load_channel_stats(dataset_info.dataset_path)

    scores_list = []

    for sample_index, (inputs, outputs, outputs_mask) in enumerate(validation_gen):
        predictions = model.predict(inputs)

        display_inputs = unscale_inputs(inputs, input_inds, scale_params)
        display_inputs = display_inputs[:, :, :, rgb_input_inds]
        display_outputs = label_to_rgb_batch(np.squeeze(outputs, axis=3))
        display_predictions = one_hot_to_rgb_batch(predictions)

        plot_prediction(
            run_path, sample_index, display_inputs, display_outputs,
            display_predictions)

        scores = compute_scores(
            outputs,
            one_hot_to_label_batch(predictions),
            outputs_mask,
            options.nb_labels)
        scores_list.append(scores)

    agg_scores = aggregate_scores(scores_list)
    save_scores(agg_scores, run_path)


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


def eval_run(model, options):
    run_path = join(results_path, options.run_name)
    dataset_info = get_dataset_info(options.dataset)

    print('Generating predictions and scores...')
    compute_predictions(
        model, dataset_info, run_path, options)

    print('Plotting graphs...')
    plot_graphs(model, run_path)
