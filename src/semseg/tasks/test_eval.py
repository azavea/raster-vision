from os.path import join

import numpy as np

from .utils import make_prediction_tile
from ..data.generators import TEST
from ..data.utils import _makedirs, save_image, zip_dir, predict_image

TEST_EVAL = 'test_eval'


def test_eval(run_path, model, options, generator):
    dataset = generator.dataset
    test_predictions_path = join(run_path, 'test_predictions')
    _makedirs(test_predictions_path)

    test_gen = generator.make_split_generator(
        TEST, tile_size=None,
        batch_size=1, shuffle=False, augment=False, normalize=True,
        eval_mode=True)

    for sample_ind, (full_tile, _, _, file_ind) in enumerate(test_gen):
        file_ind = file_ind[0]
        print('Processing {}'.format(file_ind))

        full_tile = np.squeeze(full_tile, axis=0)

        prediction_tile = make_prediction_tile(
            full_tile, options.tile_size[0],
            lambda x: dataset.one_hot_to_rgb_batch(predict_image(x, model)))

        prediction_file_path = join(
            test_predictions_path,
            generator.dataset.get_output_file_name(file_ind))
        save_image(prediction_tile, prediction_file_path)

        if (options.nb_eval_samples is not None and
                sample_ind == options.nb_eval_samples - 1):
            break

    zip_path = join(run_path, 'submission.zip')
    zip_dir(test_predictions_path, zip_path)
