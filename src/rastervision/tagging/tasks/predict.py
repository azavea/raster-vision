from os.path import join
import json

import numpy as np

from rastervision.common.settings import results_path

from rastervision.tagging.data.planet_kaggle import TagStore
from rastervision.tagging.tasks.utils import compute_prediction
from rastervision.tagging.tasks.train_thresholds import load_thresholds
from rastervision.common.settings import TRAIN, VALIDATION, TEST

TRAIN_PROBS = 'train_probs'
VALIDATION_PROBS = 'validation_probs'
TEST_PROBS = 'test_probs'

TRAIN_PREDICT = 'train_predict'
VALIDATION_PREDICT = 'validation_predict'
TEST_PREDICT = 'test_predict'

NUM_AUG = 6


def get_probs_fn(split):
    return '{}_probs.npy'.format(split)


def get_preds_fn(split):
    return '{}_preds.csv'.format(split)


def get_aug_probs_fn(split, aug_ind):
    return str(aug_ind) + '_' + get_probs_fn(split)


def check_augmentation(options, split):
    train_flag, val_flag, test_flag = False, False, False

    if options.train_augmentation and split == TRAIN:
        return True
    if options.val_augmentation and split == VALIDATION:
        return True
    if options.test_augmentation and split == TEST:
        return True

    return False


def compute_concat_probs(run_path, options, generator, split):
    probs_path = join(run_path, get_probs_fn(split))

    y_probs_list = []
    active_tags_list = []

    for run_name in options.aggregate_run_names:
        run_probs_path = join(
            results_path, run_name, get_probs_fn(split))
        options_path = join(
            results_path, run_name, 'options.json')

        with open(options_path) as options_file:
            # TODO check that active_tags are disjoint
            options_dict = json.load(options_file)
            active_tags = options_dict.get(
                'active_tags', generator.dataset.all_tags)
            active_tags_list.append(active_tags)
            y_probs_list.append(np.load(run_probs_path))

    nb_samples = y_probs_list[0].shape[0]
    nb_active_tags = sum(map(lambda x: len(x), active_tags_list))
    concat_y_probs = np.zeros((nb_samples, nb_active_tags))
    for y_probs, active_tags in zip(y_probs_list, active_tags_list):
        for tag_ind, active_tag in enumerate(active_tags):
            concat_tag_ind = generator.tag_store.get_tag_ind(active_tag)
            concat_y_probs[:, concat_tag_ind] = y_probs[:, tag_ind]

    np.save(probs_path, concat_y_probs)


def compute_ensemble_probs(run_path, options, generator, split):
    probs_path = join(run_path, get_probs_fn(split))
    y_probs = []

    for run_name in options.aggregate_run_names:
        run_probs_path = join(
            results_path, run_name, get_probs_fn(split))
        y_probs.append(np.expand_dims(np.load(run_probs_path), axis=2))

    y_probs = np.concatenate(y_probs, axis=2)
    y_probs = np.mean(y_probs, axis=2)

    if options.nb_eval_samples is not None:
        y_probs = y_probs[0:options.nb_eval_samples, :]

    np.save(probs_path, y_probs)


def compute_probs(run_path, model, options, generator, split):
    batch_size = options.batch_size
    split_gen = generator.make_split_generator(
        split, target_size=None,
        batch_size=batch_size, shuffle=False, augment_methods=None,
        normalize=True, only_xy=False)

    # Performing safe augmentations on images one by one, predicting
    # probs and taking average, if calculating probabilities
    if check_augmentation(options, split):
        y_probs = [[] for i in range(NUM_AUG)]
        mean_y_probs = []
        for batch_ind, batch in enumerate(split_gen):
            for img_ind in range(batch.x.shape[0]):
                img = batch.x[img_ind, :, :, :]
                aug_batch = []
                mean_aug_probs = []

                for rotation in range(4):
                    rot_img = np.rot90(img, rotation)
                    aug_batch.append(np.expand_dims(rot_img, axis=0))
                flip_img = np.flipud(img)
                aug_batch.append(np.expand_dims(flip_img, axis=0))
                flip_img = np.fliplr(img)
                aug_batch.append(np.expand_dims(flip_img, axis=0))
                aug_batch = np.concatenate(aug_batch, axis=0)
                aug_probs = model.predict(aug_batch)
                mean_aug_probs = np.mean(aug_probs, axis=0,
                                         keepdims=True)

                mean_y_probs.append(mean_aug_probs)
                for aug_ind in range(NUM_AUG):
                    y_probs[aug_ind].append(aug_probs[aug_ind])

            if (options.nb_eval_samples is not None and
                    batch_ind * options.batch_size >= options.nb_eval_samples):
                break

        for aug_ind in range(NUM_AUG):
            y_probs[aug_ind] = np.concatenate([y_probs[aug_ind]])
            if options.nb_eval_samples is not None:
                y_probs[aug_ind] = y_probs[aug_ind][0:options.nb_eval_samples]
            probs_path = join(run_path, get_aug_probs_fn(split, aug_ind))
            np.save(probs_path, y_probs[aug_ind])

        mean_y_probs = np.concatenate(mean_y_probs, axis=0)
        if options.nb_eval_samples is not None:
            mean_y_probs = mean_y_probs[0:options.nb_eval_samples, :]
        probs_path = join(run_path, get_probs_fn(split))
        np.save(probs_path, mean_y_probs)
    # Otherwise, predicting probs on unaugmented images by batch
    else:
        y_probs = []
        for batch_ind, batch in enumerate(split_gen):
            y_probs.append(model.predict(batch.x))

            if (options.nb_eval_samples is not None and
                    batch_ind * options.batch_size >= options.nb_eval_samples):
                break

        y_probs = np.concatenate(y_probs, axis=0)

        if options.nb_eval_samples is not None:
            y_probs = y_probs[0:options.nb_eval_samples, :]

        probs_path = join(run_path, get_probs_fn(split))
        np.save(probs_path, y_probs)


def compute_preds(run_path, options, generator, split):
    thresholds = load_thresholds(run_path, options.loss_function)

    if check_augmentation(options, split):
        tag_stores = []
        file_inds = generator.get_file_inds(split)

        y_probs = []
        for aug_ind in range(NUM_AUG):
            tag_store = TagStore(active_tags=options.active_tags)
            tag_stores.append(tag_store)

        for aug_ind in range(NUM_AUG):
            probs_path = join(run_path, get_aug_probs_fn(split, aug_ind))
            y_probs = np.load(probs_path)

            for sample_ind in range(y_probs.shape[0]):
                y_pred = compute_prediction(
                    y_probs[sample_ind, :], generator.dataset,
                    generator.tag_store, thresholds)
                file_ind = file_inds[sample_ind]
                tag_stores[aug_ind].add_tags(file_ind, y_pred)

            predictions_path = join(run_path, str(aug_ind) + '_' +
                                    get_preds_fn(split))
            tag_stores[aug_ind].save(predictions_path)

    probs_path = join(run_path, get_probs_fn(split))
    y_probs = np.load(probs_path)
    predictions_path = join(run_path, get_preds_fn(split))
    tag_store = TagStore(active_tags=options.active_tags)
    file_inds = generator.get_file_inds(split)

    for sample_ind in range(y_probs.shape[0]):
        y_pred = compute_prediction(
            y_probs[sample_ind, :], generator.dataset, generator.tag_store,
            thresholds)
        file_ind = file_inds[sample_ind]
        tag_store.add_tags(file_ind, y_pred)

    tag_store.save(predictions_path)
