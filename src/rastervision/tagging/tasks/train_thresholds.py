from os.path import join

from sklearn.metrics import fbeta_score
import numpy as np

from rastervision.common.settings import TRAIN
from rastervision.common.utils import save_json, load_json

TRAIN_THRESHOLDS = 'train_thresholds'


def load_thresholds(run_path):
    return np.array(load_json(join(run_path, 'thresholds.json')))


def save_thresholds(run_path, thresholds):
    save_json(thresholds.tolist(), join(run_path, 'thresholds.json'))


def get_model_output(run_path, generator, nb_eval_samples=None):
    split = TRAIN
    file_inds = generator.get_file_inds(split)
    if nb_eval_samples is not None:
        file_inds = file_inds[0:nb_eval_samples]
    y_true = generator.tag_store.get_tag_array(file_inds)

    probs_path = join(run_path, '{}_probs.npy'.format(split))
    y_probs = np.load(probs_path)
    if nb_eval_samples is not None:
        y_probs = y_probs[0:nb_eval_samples, :]

    return y_true, y_probs


def optimize_thresholds(y_true, y_probs):
    nb_tags = y_true.shape[1]
    best_thresholds = np.ones((nb_tags,)) * 0.2
    y_preds = y_probs > np.expand_dims(best_thresholds, axis=0)
    best_f2 = fbeta_score(y_true, y_preds, beta=2, average='samples')

    for tag_ind in range(nb_tags):
        thresholds = np.copy(best_thresholds)
        for tag_thresh in np.arange(0, 1.0, 0.02):
            thresholds[tag_ind] = tag_thresh
            y_preds = y_probs > np.expand_dims(thresholds, axis=0)
            f2 = fbeta_score(y_true, y_preds, beta=2, average='samples')
            if f2 > best_f2:
                best_f2 = f2
                best_thresholds = np.copy(thresholds)

    return best_thresholds


def train_thresholds(run_path, options, generator):
    # Finding the correct thresholds seems to result in a small decrease in
    # test performance. I'm not sure what's going on, so for now
    # I'm commenting this out and setting thresholds to a default of 0.2 to
    # maintain the previous performance.

    y_true, y_probs = get_model_output(
        run_path, generator, options.nb_eval_samples)
    # thresholds = optimize_thresholds(y_true, y_probs)
    thresholds = 0.2 * np.ones((len(generator.tag_store.active_tags),))
    save_thresholds(run_path, thresholds)
