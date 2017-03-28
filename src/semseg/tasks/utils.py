"""Utility functions shared across tasks."""
import numpy as np


def make_prediction_img(full_img, target_size, predict):
    quarter_target_size = target_size // 4
    half_target_size = target_size // 2
    nb_prediction_channels = \
        predict(full_img[0:target_size, 0:target_size, :]).shape[2]
    full_prediction_img = np.zeros(
        (full_img.shape[0], full_img.shape[1], nb_prediction_channels),
        dtype=np.uint8)

    def snap_bounds(row_begin, row_end, col_begin, col_end):
        # If the img straddles the edge of the full_img, then
        # snap it to the edge.
        if row_end > full_img.shape[0]:
            row_begin = full_img.shape[0] - target_size
            row_end = full_img.shape[0]

        if col_end > full_img.shape[1]:
            col_begin = full_img.shape[1] - target_size
            col_end = full_img.shape[1]

        return row_begin, row_end, col_begin, col_end

    def update_prediction(row_begin, row_end, col_begin, col_end):
        row_begin, row_end, col_begin, col_end = \
            snap_bounds(row_begin, row_end, col_begin, col_end)

        img = full_img[row_begin:row_end, col_begin:col_end, :]
        prediction_img = predict(img)

        full_prediction_img[row_begin:row_end, col_begin:col_end, :] = \
            prediction_img

    def update_prediction_crop(row_begin, row_end, col_begin, col_end):
        row_begin, row_end, col_begin, col_end = \
            snap_bounds(row_begin, row_end, col_begin, col_end)

        img = full_img[row_begin:row_end, col_begin:col_end, :]
        prediction_img = predict(img)

        prediction_img_crop = prediction_img[
            quarter_target_size:target_size - quarter_target_size,
            quarter_target_size:target_size - quarter_target_size,
            :]

        full_prediction_img[
            row_begin + quarter_target_size:row_end - quarter_target_size,
            col_begin + quarter_target_size:col_end - quarter_target_size,
            :] = prediction_img_crop

    for row_begin in range(0, full_img.shape[0], half_target_size):
        for col_begin in range(0, full_img.shape[1], half_target_size):
            row_end = row_begin + target_size
            col_end = col_begin + target_size

            is_edge = (row_begin == 0 or row_end >= full_img.shape[0] or
                       col_begin == 0 or col_end >= full_img.shape[1])

            if is_edge:
                update_prediction(row_begin, row_end, col_begin, col_end)
            else:
                update_prediction_crop(row_begin, row_end, col_begin, col_end)

    return full_prediction_img
