from os.path import join

import numpy as np

from rastervision.common.settings import VALIDATION, TEST
from rastervision.tagging.data.planet_kaggle import TagStore

VALIDATION_PREDICT = 'validation_predict'
TEST_PREDICT = 'test_predict'


def compute_prediction(y_probs, dataset):
    atmos_inds = [dataset.get_tag_ind(tag)
                  for tag in dataset.atmos_tags]

    # TODO experimenting with a different threshold
    # remove this after the loss function is a better approximate
    # of f2
    decision_thresh = 0.2
    y_pred = (y_probs > decision_thresh).astype(np.float32)

    # TODO remove this post-processing step once our model
    # enforces the constraint that there is at least one atmospheric
    # tag.
    if np.sum(y_pred[atmos_inds]) == 0:
        max_ind = np.argmax(y_probs[atmos_inds])
        max_tag = dataset.atmos_tags[max_ind]
        max_ind = dataset.get_tag_ind(max_tag)
        y_pred[max_ind] = 1

    return y_pred


def predict(run_path, model, options, generator, split):
    """Generate predictions for split data.

    For each image in a split, create a prediction, add it to the TagStore,
    and then save the TagStore to a csv file.

    # Arguments
        run_path: the path to the files for a run
        model: a Keras model that has been trained
        options: RunOptions object that specifies the run
        generator: a Generator object to generate the test data
        split: name of the split eg. validation
    """
    predictions_path = join(run_path, '{}_predictions.csv'.format(split))

    batch_size = options.batch_size
    split_gen = generator.make_split_generator(
        split, target_size=None,
        batch_size=batch_size, shuffle=False, augment=False,
        normalize=True, only_xy=False)

    tag_store = TagStore()

    for batch_ind, batch in enumerate(split_gen):
        y_probs = model.predict(batch.x)
        for sample_ind in range(batch.x.shape[0]):
            y_pred = compute_prediction(
                y_probs[sample_ind, :], generator.dataset)
            tag_store.add_tags(
                batch.file_inds[sample_ind], y_pred)

        if (options.nb_eval_samples is not None and
                batch_ind * options.batch_size >= options.nb_eval_samples):
            break

    tag_store.save(predictions_path)


def validation_predict(run_path, model, options, generator):
    predict(run_path, model, options, generator, VALIDATION)


def test_predict(run_path, model, options, generator):
    predict(run_path, model, options, generator, TEST)
