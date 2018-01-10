import json
from os.path import join, dirname

import click
import numpy as np
import rasterio

from rv.utils.files import make_dir
from rv.utils.geo import load_window
from rv.utils.misc import save_img
from rv.detection.commands.settings import default_channel_order


def _make_predict_chips(image_path, chips_dir, chips_info_path,
                        chip_size=300, channel_order=default_channel_order):
    """Slide window (with overlap) over image to generate prediction chips.

    The neural network can only make predictions over small, fixed sized images
    so this breaks a large image into a set of chips to feed into the network.
    """
    click.echo('Making predict chips...')
    make_dir(chips_dir, check_empty=True)
    make_dir(chips_info_path, use_dirname=True)
    image_dataset = rasterio.open(image_path)

    offsets = {}
    # Add padding to some chips to account for image not being divisible by
    # the chip_size.
    for row_start in range(0, image_dataset.height, chip_size // 2):
        row_end = min(row_start + chip_size, image_dataset.height)
        for col_start in range(0, image_dataset.width, chip_size // 2):
            col_end = min(col_start + chip_size, image_dataset.width)

            chip = load_window(
                image_dataset, channel_order,
                window=((row_start, row_end), (col_start, col_end)))
            padded_chip = np.zeros(
                (chip_size, chip_size, 3))
            padded_chip[0:chip.shape[0], 0:chip.shape[1], :] = chip

            chip_filename = '{}_{}.png'.format(row_start, col_start)
            chip_path = join(chips_dir, chip_filename)
            save_img(chip_path, padded_chip)

            # Position of the upper-left corner of chip in the
            # original, unpadded image.
            offsets[chip_filename] = (col_start, row_start)

    chips_info = {
        'offsets': offsets,
        'chip_size': chip_size
    }
    with open(chips_info_path, 'w') as chips_info_file:
        json.dump(chips_info, chips_info_file)


@click.command()
@click.argument('image_path')
@click.argument('chips_dir')
@click.argument('chips_info_path')
@click.option('--chip-size', default=300,
              help='Height and width of each chip')
@click.option('--channel-order', nargs=3, type=int,
              default=default_channel_order, help='Indices of RGB channels')
def make_predict_chips(image_path, chips_dir, chips_info_path,
                       chip_size, channel_order):
    """Generate chips from large images to run prediction on.

    The output contains chips and a JSON file with the (x, y) coordinates
    in pixels of the upper-left corner of each chip in the frame of reference
    of the original image.

    Args:
        image_path: TIFF or VRT file to create chips from
        chips_dir: Directory for chip files
        chips_info_path: Chips info file
    """
    _make_predict_chips(image_path, chips_dir, chips_info_path,
                        chip_size=chip_size, channel_order=channel_order)


if __name__ == '__main__':
    make_predict_chips()
