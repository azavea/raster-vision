import json
import argparse
from os import makedirs
from os.path import join

import numpy as np
# from scipy.ndimage import imread
from scipy.misc import imsave

from utils import load_tiff


def pad_image(im, window_size):
    '''Pad image so it's divisible by window_size.'''
    row_pad = window_size - (im.shape[0] % window_size)
    col_pad = window_size - (im.shape[1] % window_size)
    row_left_pad = row_pad // 2
    col_left_pad = col_pad // 2

    pad_width = (
        (row_left_pad, row_pad - row_left_pad),
        (col_left_pad, col_pad - col_left_pad),
        (0, 0)
    )
    pad_im = np.pad(im, pad_width, mode='constant')
    return pad_im, row_left_pad, col_left_pad


def make_windows(image_path, output_dir, window_size):
    images_dir = join(output_dir, 'images')
    makedirs(images_dir, exist_ok=True)

    im, _ = load_tiff(image_path)
    pad_im, row_left_pad, col_left_pad = pad_image(im, window_size)

    offsets = {}
    for i in range(0, pad_im.shape[0], window_size // 2):
        for j in range(0, pad_im.shape[1], window_size // 2):
            if (i + window_size > pad_im.shape[0] or
                    j + window_size > pad_im.shape[1]):
                break

            window = pad_im[i:i+window_size, j:j+window_size, :]
            window_file_name = '{}_{}.png'.format(i, j)
            window_path = join(images_dir, window_file_name)
            imsave(window_path, window)

            # Position of the upper-left corner of window in the
            # original, unpadded image.
            offsets[window_file_name] = \
                (j - col_left_pad, i - row_left_pad)

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
