from os.path import join
from shutil import rmtree

import numpy as np

from rastervision.common.utils import _makedirs, save_img, zip_dir
from rastervision.common.settings import VALIDATION, TEST

from rastervision.semseg.tasks.utils import make_prediction_img, predict_x


VALIDATION_PREDICT = 'validation_predict'
TEST_PREDICT = 'test_predict'


def predict(run_path, model, options, generator, split, save_probs=False):
    """Generate predictions for split data.

    For each image in a split, create a prediction image .tif file, and then
    zip them into a zip file. Do the same for the predicted probability images.

    # Arguments
        run_path: the path to the files for a run
        model: a Keras model that has been trained
        options: RunOptions object that specifies the run
        generator: a Generator object to generate the test data
        split: name of the split eg. validation
    """
    dataset = generator.dataset
    if save_probs:
        probs_path = join(run_path, '{}_probs'.format(split))
        _makedirs(probs_path)
    predictions_path = join(run_path, '{}_predictions'.format(split))
    _makedirs(predictions_path)

    split_gen = generator.make_split_generator(
        split, target_size=options.eval_target_size,
        batch_size=1, shuffle=False, augment_methods=None, normalize=True,
        only_xy=False)

    for sample_ind, batch in enumerate(split_gen):
        file_ind = batch.file_inds[0]
        print('Processing {}'.format(file_ind))

        x = np.squeeze(batch.x, axis=0)

        y_probs = make_prediction_img(
            x, options.target_size[0],
            lambda x: predict_x(x, model))

        if save_probs:
            probs_file_path = join(
                probs_path,
                generator.dataset.get_output_file_name(file_ind))
            save_img(y_probs, probs_file_path)

        y_preds = dataset.one_hot_to_rgb_batch(y_probs)
        prediction_file_path = join(
            predictions_path,
            generator.dataset.get_output_file_name(file_ind))
        save_img(y_preds, prediction_file_path)

        if (options.nb_eval_samples is not None and
                sample_ind == options.nb_eval_samples - 1):
            break

    if save_probs:
        zip_path = join(run_path, '{}_probs.zip'.format(split))
        zip_dir(probs_path, zip_path)
        rmtree(probs_path)

    zip_path = join(run_path, '{}_predictions.zip'.format(split))
    zip_dir(predictions_path, zip_path)
    rmtree(predictions_path)


def validation_predict(run_path, model, options, generator):
    predict(run_path, model, options, generator, VALIDATION, save_probs=True)


def test_predict(run_path, model, options, generator):
    predict(run_path, model, options, generator, TEST)
