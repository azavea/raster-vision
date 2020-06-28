import os
import glob

import rasterio
from PIL import Image
import numpy as np
import click

from object_detection.utils.np_box_list import BoxList

from rv.utils import save_geojson, make_empty_dir


def png_to_geojson(geotiff_path, label_png_path, output_path, object_half_len):
    """Convert COWC PNG labels to GeoJSON format.
    In the COWC dataset, the center position of cars is represented as
    non-zero pixels in PNG files that are aligned with the GeoTIFFs.
    This script converts the PNG file to a GeoJSON representation.
    """
    image_dataset = rasterio.open(geotiff_path)
    label_im = np.array(Image.open(label_png_path))

    point_inds = np.argwhere(label_im[:, :, 0] != 0).astype(np.float)

    # Normalize inds
    point_inds[:, 0] /= label_im.shape[0]
    point_inds[:, 1] /= label_im.shape[1]

    # Convert to geotiff image inds
    point_inds[:, 0] *= image_dataset.height
    point_inds[:, 1] *= image_dataset.width
    point_inds = point_inds.astype(np.int)

    # Turn points into squares and ensure edges aren't outside the array
    y_min = np.clip(point_inds[:, 0:1] - object_half_len, 0,
                    image_dataset.height)
    x_min = np.clip(point_inds[:, 1:2] - object_half_len, 0,
                    image_dataset.width)
    y_max = np.clip(point_inds[:, 0:1] + object_half_len, 0,
                    image_dataset.height)
    x_max = np.clip(point_inds[:, 1:2] + object_half_len, 0,
                    image_dataset.width)

    # Write to GeoJSON
    boxes = np.hstack([y_min, x_min, y_max, x_max]).astype(np.float)
    boxlist = BoxList(boxes)
    save_geojson(output_path, boxlist, image_dataset=image_dataset)
    return boxlist


@click.command()
@click.argument('geotiff_dir')
@click.argument('label_png_dir')
@click.argument('output_dir')
@click.option('--object-half-len', default=50)
def prepare_potsdam(geotiff_dir, label_png_dir, output_dir, object_half_len):
    label_paths = glob.glob(
        os.path.join(label_png_dir, 'top_potsdam_*_RGB_Annotated_Cars.png'))
    make_empty_dir(output_dir)

    for label_path in label_paths:
        geotiff_base = os.path.basename(label_path)[0:-19]
        geotiff_path = os.path.join(geotiff_dir, geotiff_base + 'IR.tif')
        output_path = os.path.join(output_dir, geotiff_base + 'IR.json')
        boxlist = png_to_geojson(
            geotiff_path,
            label_path,
            output_path,
            object_half_len=object_half_len)
        print('Saved {} with {} boxes.'.format(output_path,
                                               boxlist.num_boxes()))


if __name__ == '__main__':
    prepare_potsdam()
