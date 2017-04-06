from os.path import join

import numpy as np

from ..data.generators import VALIDATION, TEST
from .utils import make_prediction_img
from ..data.utils import _makedirs, save_img, zip_dir, predict_img

VALIDATION_PREDICT = 'validation_predict'
TEST_PREDICT = 'test_predict'


def predict(run_path, model, options, generator, split):
    """Generate predictions for split data.

    For each image in a split, create a prediction image .tif file, and then
    zip them into a zip file.

    # Arguments
        run_path: the path to the files for a run
        model: a Keras model that has been trained
        options: RunOptions object that specifies the run
        generator: a Generator object to generate the test data
        split: name of the split eg. validation
    """
    dataset = generator.dataset
    predictions_path = join(run_path, '{}_predictions'.format(split))
    _makedirs(predictions_path)

    split_gen = generator.make_split_generator(
        split, target_size=None,
        batch_size=1, shuffle=False, augment=False, normalize=True,
        eval_mode=True)

    for sample_ind, (batch_x, _, _, _, file_ind) in enumerate(split_gen):
        file_ind = file_ind[0]
        print('Processing {}'.format(file_ind))

        x = np.squeeze(batch_x, axis=0)

        y = make_prediction_img(
            x, options.target_size[0],
            lambda x: dataset.one_hot_to_rgb_batch(predict_img(x, model)))

        prediction_file_path = join(
            predictions_path,
            generator.dataset.get_output_file_name(file_ind))
        save_img(y, prediction_file_path)

        if (options.nb_eval_samples is not None and
                sample_ind == options.nb_eval_samples - 1):
            break

    zip_path = join(run_path, '{}_predictions.zip'.format(split))
    zip_dir(predictions_path, zip_path)


def validation_predict(run_path, model, options, generator):
    predict(run_path, model, options, generator, VALIDATION)


def test_predict(run_path, model, options, generator):
    predict(run_path, model, options, generator, TEST)
