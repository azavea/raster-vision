from os.path import join

import numpy as np

from .utils import make_prediction_img
from ..data.generators import TEST
from ..data.utils import _makedirs, save_img, zip_dir, predict_img

TEST_EVAL = 'test_eval'


def test_eval(run_path, model, options, generator):
    """Generate predictions for test data.

    For each test image, create a prediction image .tif file, and then zip
    them into submission.zip.

    # Arguments
        run_path: the path to the files for a run
        model: a Keras model that has been trained
        options: RunOptions object that specifies the run
        generator: a Generator object to generate the test data
    """
    dataset = generator.dataset
    test_predictions_path = join(run_path, 'test_predictions')
    _makedirs(test_predictions_path)

    test_gen = generator.make_split_generator(
        TEST, target_size=None,
        batch_size=1, shuffle=False, augment=False, normalize=True,
        eval_mode=True)

    for sample_ind, (batch_x, _, _, _, file_ind) in enumerate(test_gen):
        file_ind = file_ind[0]
        print('Processing {}'.format(file_ind))

        x = np.squeeze(batch_x, axis=0)

        y = make_prediction_img(
            x, options.target_size[0],
            lambda x: dataset.one_hot_to_rgb_batch(predict_img(x, model)))

        prediction_file_path = join(
            test_predictions_path,
            generator.dataset.get_output_file_name(file_ind))
        save_img(y, prediction_file_path)

        if (options.nb_eval_samples is not None and
                sample_ind == options.nb_eval_samples - 1):
            break

    zip_path = join(run_path, 'submission.zip')
    zip_dir(test_predictions_path, zip_path)
