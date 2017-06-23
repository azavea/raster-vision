from os.path import join

import numpy as np

from rastervision.common.settings import results_path

from rastervision.tagging.data.planet_kaggle import TagStore
from rastervision.tagging.tasks.utils import compute_prediction
from rastervision.tagging.tasks.train_thresholds import load_thresholds

TRAIN_PROBS = 'train_probs'
VALIDATION_PROBS = 'validation_probs'
TEST_PROBS = 'test_probs'

TRAIN_PREDICT = 'train_predict'
VALIDATION_PREDICT = 'validation_predict'
TEST_PREDICT = 'test_predict'


def compute_ensemble_probs(run_path, options, generator, split):
    probs_path = join(run_path, '{}_probs.npy'.format(split))
    y_probs = []

    for run_name in options.aggregate_run_names:
        run_probs_path = join(
            results_path, run_name, '{}_probs.npy'.format(split))
        y_probs.append(np.expand_dims(np.load(run_probs_path), axis=2))

    y_probs = np.concatenate(y_probs, axis=2)
    y_probs = np.mean(y_probs, axis=2)

    if options.nb_eval_samples is not None:
        y_probs = y_probs[0:options.nb_eval_samples, :]

    np.save(probs_path, y_probs)


def compute_probs(run_path, model, options, generator, split):
    probs_path = join(run_path, '{}_probs.npy'.format(split))
    y_probs = []

    batch_size = options.batch_size
    split_gen = generator.make_split_generator(
        split, target_size=None,
        batch_size=batch_size, shuffle=False, augment_methods=None,
        normalize=True, only_xy=False)

    for batch_ind, batch in enumerate(split_gen):
        y_probs.append(model.predict(batch.x))

        if (options.nb_eval_samples is not None and
                batch_ind * options.batch_size >= options.nb_eval_samples):
            break

    y_probs = np.concatenate(y_probs, axis=0)
    if options.nb_eval_samples is not None:
        y_probs = y_probs[0:options.nb_eval_samples, :]

    np.save(probs_path, y_probs)


def compute_preds(run_path, options, generator, split):
    probs_path = join(run_path, '{}_probs.npy'.format(split))
    y_probs = np.load(probs_path)

    predictions_path = join(run_path, '{}_preds.csv'.format(split))
    thresholds = load_thresholds(run_path)
    tag_store = TagStore(active_tags=options.active_tags)
    file_inds = generator.get_file_inds(split)

    for sample_ind in range(y_probs.shape[0]):
        y_pred = compute_prediction(
            y_probs[sample_ind, :], generator.dataset, generator.tag_store,
            thresholds)
        file_ind = file_inds[sample_ind]
        tag_store.add_tags(file_ind, y_pred)

    tag_store.save(predictions_path)
