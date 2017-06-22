from os.path import join
import json

import numpy as np
from sklearn import metrics

from rastervision.common.utils import _makedirs, safe_divide
from rastervision.common.settings import VALIDATION

from rastervision.semseg.tasks.utils import (
    predict_x, make_prediction_img, plot_prediction)

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


def save_scores(scores, run_path):
    with open(join(run_path, 'scores.json'), 'w') as scores_file:
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
        batch_size=1, shuffle=False, augment_methods=None, normalize=True,
        only_xy=False)

    confusion_mat = np.zeros((dataset.nb_labels, dataset.nb_labels))
    predictions_path = join(run_path, 'validation_eval')
    _makedirs(predictions_path)

    for sample_index, batch in enumerate(validation_gen):
        file_ind = batch.file_inds[0]
        print('Processing {}'.format(file_ind))

        x = np.squeeze(batch.x, axis=0)
        all_x = np.squeeze(batch.all_x, axis=0)
        y = np.squeeze(batch.y, axis=0)
        y_mask = np.squeeze(batch.y_mask, axis=0)

        display_pred = make_prediction_img(
            x, options.target_size[0],
            lambda x: dataset.one_hot_to_rgb_batch(predict_x(x, model)))
        display_y = dataset.one_hot_to_rgb_batch(y)

        label_y = dataset.one_hot_to_label_batch(y)
        label_pred = dataset.rgb_to_label_batch(display_pred)

        confusion_mat += compute_confusion_mat(
            label_y, y_mask, label_pred, dataset.nb_labels)

        if (options.nb_eval_plot_samples is not None and
                sample_index < options.nb_eval_plot_samples):
            file_path = '{}.png'.format(sample_index)
            file_path = join(predictions_path, file_path)
            plot_prediction(
                generator, all_x, display_y, display_pred, file_path,
                is_debug=True)

        if (options.nb_eval_samples is not None and
                sample_index == options.nb_eval_samples - 1):
            break

    scores = Scores()
    scores.compute_scores(label_names, confusion_mat)
    save_scores(scores, run_path)
