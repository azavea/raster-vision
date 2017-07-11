from os.path import join

from sklearn.metrics import fbeta_score
import numpy as np

from rastervision.common.settings import TRAIN
from rastervision.common.utils import save_json, load_json
from rastervision.tagging.tasks.utils import compute_prediction

TRAIN_THRESHOLDS = 'train_thresholds'
BINARY_CROSSENTROPY = 'binary_crossentropy'


def load_thresholds(run_path, loss_function):
    if loss_function == BINARY_CROSSENTROPY:
        return np.array(load_json(join(run_path, 'thresholds.json')))
    return None


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


def optimize_thresholds(y_true, y_probs, tag_store, dataset):
    nb_tags = y_true.shape[1]
    best_thresholds = np.ones((nb_tags,)) * 0.2
    thresh_inc = 0.03

    y_preds = compute_prediction(y_probs, dataset, tag_store, best_thresholds)
    best_f2 = fbeta_score(y_true, y_preds, beta=2, average='samples')
    print(best_f2)

    for tag in tag_store.active_tags:
        tag_ind = tag_store.get_tag_ind(tag)
        thresholds = np.copy(best_thresholds)
        for tag_thresh in np.arange(0, 1.0, thresh_inc):
            thresholds[tag_ind] = tag_thresh
            y_preds = compute_prediction(
                y_probs, dataset, tag_store, thresholds)
            f2 = fbeta_score(y_true, y_preds, beta=2, average='samples')
            if f2 > best_f2:
                print('tag: {:>20}, thresh: {:>10}, f2: {:>10.5}'.format(
                    tag, tag_thresh, f2))
                best_f2 = f2
                best_thresholds = np.copy(thresholds)

    return best_thresholds


def train_thresholds(run_path, options, generator):
    if options.loss_function == BINARY_CROSSENTROPY:
        y_true, y_probs = get_model_output(
            run_path, generator, options.nb_eval_samples)
        thresholds = optimize_thresholds(
            y_true, y_probs, generator.tag_store, generator.dataset)
        save_thresholds(run_path, thresholds)
