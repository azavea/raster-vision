import json
import argparse
from os import makedirs
from os.path import join

import numpy as np
# from scipy.ndimage import imread
from scipy.misc import imsave
import rasterio

from utils import load_window


def make_windows(image_path, output_dir, window_size):
    images_dir = join(output_dir, 'images')
    makedirs(images_dir, exist_ok=True)
    image_dataset = rasterio.open(image_path)

    offsets = {}
    for row_start in range(0, image_dataset.height, window_size // 2):
        row_end = min(row_start + window_size, image_dataset.height)
        for col_start in range(0, image_dataset.width, window_size // 2):
            col_end = min(col_start + window_size, image_dataset.width)

            window = load_window(
                image_dataset,
                window=((row_start, row_end), (col_start, col_end)))
            padded_window = np.zeros(
                (image_dataset.height, image_dataset.width, 3))
            padded_window[row_start:row_end, col_start:col_end, :] = window

            window_file_name = '{}_{}.png'.format(row_start, col_start)
            window_path = join(images_dir, window_file_name)
            imsave(window_path, window)

            # Position of the upper-left corner of window in the
            # original, unpadded image.
            offsets[window_file_name] = (col_start, row_start)

    window_info = {
        'offsets': offsets,
        'window_size': window_size
    }
    window_info_path = join(output_dir, 'window_info.json')
    with open(window_info_path, 'w') as window_info_file:
        json.dump(window_info, window_info_file)


def parse_args():
    description = """
        Slide window over image and generate small window image files.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--image-path')
    parser.add_argument('--output-dir')
    parser.add_argument('--window-size', type=int, default=300)

    return parser.parse_args()


if __name__ == '__main__':
        args = parse_args()
        print(args)

        make_windows(args.image_path, args.output_dir, args.window_size)
