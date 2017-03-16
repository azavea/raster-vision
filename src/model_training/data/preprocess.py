"""
Process data from the 2D semantic labeling for aerial imagery dataset
http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html
to put it into a Keras-friendly format.
"""
from os.path import join

import numpy as np

from .settings import (
    POTSDAM, get_dataset_info)
from .utils import (
    _makedirs, load_tiff, load_image, rgb_to_label_batch, save_image,
    rgb_to_mask, compute_ndvi)
from .generators import save_channel_stats, get_channel_stats


def process_data(file_inds, label_keys, raw_rgbir_input_path, raw_depth_input_path,
                 proc_data_path, get_file_names, raw_output_path=None,
                 raw_output_mask_path=None):
    print('Processing data...')
    _makedirs(proc_data_path)

    for index1, index2 in file_inds:
        print('{}_{}'.format(index1, index2))

        (rgbir_file_name, depth_file_name, output_file_name,
         output_mask_file_name) = get_file_names(index1, index2)

        rgbir_input_im = load_tiff(join(raw_rgbir_input_path, rgbir_file_name))

        depth_input_im = load_image(
            join(raw_depth_input_path, depth_file_name))
        depth_input_im = np.expand_dims(depth_input_im, axis=2)

        red = rgbir_input_im[:, :, 0]
        ir = rgbir_input_im[:, :, 3]
        ndvi_im = compute_ndvi(red, ir)
        # NDVI ranges from [-1.0, 1.0]. We need to make this value fit into a
        # uint8 so we scale it.
        ndvi_im = (ndvi_im + 1) * 127

        if raw_output_path and raw_output_mask_path:
            output_im = load_tiff(join(raw_output_path, output_file_name))
            output_im = np.expand_dims(output_im, axis=0)
            output_im = rgb_to_label_batch(output_im, label_keys)
            output_im = np.squeeze(output_im, axis=0)
            output_im = np.expand_dims(output_im, axis=2)

            output_mask_im = load_tiff(
                join(raw_output_mask_path, output_mask_file_name))
            output_mask_im = rgb_to_mask(output_mask_im)
            output_mask_im = np.expand_dims(output_mask_im, axis=2)

            concat_im = np.concatenate(
                [rgbir_input_im, depth_input_im, ndvi_im, output_im,
                 output_mask_im], axis=2)
        else:
            # There's one depth image that's missing a column,
            # so we add a column with zeros.
            if depth_input_im.shape[1] == 5999:
                depth_input_im_fix = np.zeros((6000, 6000, 1), dtype=np.uint8)
                depth_input_im_fix[:, 0:-1, :] = depth_input_im
                depth_input_im = depth_input_im_fix

            concat_im = np.concatenate(
                [rgbir_input_im, depth_input_im, ndvi_im], axis=2)

        proc_file_name = '{}_{}'.format(index1, index2)
        save_image(join(proc_data_path, proc_file_name), concat_im)


def process_potsdam():
    dataset_info = get_dataset_info(POTSDAM)
    proc_data_path = dataset_info.dataset_path
    raw_data_path = dataset_info.raw_dataset_path

    raw_rgbir_input_path = join(raw_data_path, '4_Ortho_RGBIR')
    raw_depth_input_path = join(raw_data_path, '1_DSM_normalisation')
    raw_output_path = join(raw_data_path, '5_Labels_for_participants')
    raw_output_mask_path = join(
        raw_data_path, '5_Labels_for_participants_no_Boundary')

    def get_file_names(index1, index2):
        rgbir_file_name = 'top_potsdam_{}_{}_RGBIR.tif'.format(index1, index2)
        depth_file_name = \
            'dsm_potsdam_{:0>2}_{:0>2}_normalized_lastools.jpg' \
            .format(index1, index2)
        output_file_name = 'top_potsdam_{}_{}_label.tif'.format(index1, index2)
        output_mask_file_name = \
            'top_potsdam_{}_{}_label_noBoundary.tif'.format(index1, index2)

        return (rgbir_file_name, depth_file_name, output_file_name,
                output_mask_file_name)

    process_data(
        dataset_info.file_inds, dataset_info.label_keys, raw_rgbir_input_path,
        raw_depth_input_path, proc_data_path, get_file_names,
        raw_output_path, raw_output_mask_path)

    process_data(
        dataset_info.test_file_inds, dataset_info.label_keys,
        raw_rgbir_input_path, raw_depth_input_path, proc_data_path,
        get_file_names)

    means, stds = get_channel_stats(
        proc_data_path, dataset_info.file_names)

    # The NDVI values are in [-1,1] by definition, but we store them as uint8s
    # in [0, 255]. So, we use a hard coded scaling for this channel to make the
    # values go back to [-1, 1], since they are more easily interpreted that
    # way and fall into the range we want for the neural network.
    ndvi_ind = dataset_info.ndvi_ind
    means[ndvi_ind] = 1.0
    stds[ndvi_ind] = 127.0
    save_channel_stats(proc_data_path, means, stds)


if __name__ == '__main__':
    process_potsdam()
