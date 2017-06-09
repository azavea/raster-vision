from os.path import join

import numpy as np

from rastervision.common.settings import VALIDATION, TEST

from rastervision.tagging.data.planet_kaggle import TagStore
from rastervision.tagging.tasks.utils import compute_prediction

VALIDATION_PREDICT = 'validation_predict'
TEST_PREDICT = 'test_predict'


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
        batch_size=batch_size, shuffle=False, augment_methods=None,
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
