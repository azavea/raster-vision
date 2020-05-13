import random

import click
import numpy as np
import rasterio
from rasterio.transform import from_origin

from rastervision.core.box import Box
from rastervision.data import (RasterioCRSTransformer, ObjectDetectionLabels,
                               ObjectDetectionGeoJSONStore)
from rastervision.core.class_map import (ClassItem, ClassMap)


@click.command()
@click.option(
    '--task',
    '-t',
    type=click.Choice(['object_detection', 'semantic_segmentation']),
    required=True)
@click.option('--chip_size', '-c', default=300, type=int)
@click.option('--chips_per_dimension', '-s', default=3, type=int)
@click.argument('tiff_path')
@click.argument('labels_path')
def generate_scene(task, tiff_path, labels_path, chip_size,
                   chips_per_dimension):
    """Generate a synthetic object detection scene.

    Randomly generates a GeoTIFF with red and greed boxes denoting two
    classes and a corresponding label file. This is useful for generating
    synthetic scenes for testing purposes.
    """
    class_map = ClassMap([ClassItem(1, 'car'), ClassItem(2, 'building')])

    # make extent that's divisible by chip_size
    chip_size = chip_size
    ymax = chip_size * chips_per_dimension
    xmax = chip_size * chips_per_dimension
    extent = Box(0, 0, ymax, xmax)

    # make windows along grid
    windows = extent.get_windows(chip_size, chip_size)

    # for each window, make some random boxes within it and render to image
    nb_channels = 3
    image = np.zeros((ymax, xmax, nb_channels)).astype(np.uint8)
    boxes = []
    class_ids = []
    for window in windows:
        # leave some windows blank
        if random.uniform(0, 1) > 0.3:
            # pick a random class
            class_id = random.randint(1, 2)
            box = window.make_random_square(50).to_int()

            boxes.append(box)
            class_ids.append(class_id)

            image[box.ymin:box.ymax, box.xmin:box.xmax, class_id - 1] = 255

    # save image as geotiff centered in philly
    transform = from_origin(-75.163506, 39.952536, 0.000001, 0.000001)

    print('Generated {} boxes with {} different classes.'.format(
        len(boxes), len(set(class_ids))))

    with rasterio.open(
            tiff_path,
            'w',
            driver='GTiff',
            height=ymax,
            transform=transform,
            crs='EPSG:4326',
            compression=rasterio.enums.Compression.none,
            width=xmax,
            count=nb_channels,
            dtype='uint8') as dst:
        for channel_ind in range(0, nb_channels):
            dst.write(image[:, :, channel_ind], channel_ind + 1)

    if task == 'object_detection':
        # make OD labels and make boxes
        npboxes = Box.to_npboxes(boxes)
        class_ids = np.array(class_ids)
        labels = ObjectDetectionLabels(npboxes, class_ids)

        # save labels to geojson
        with rasterio.open(tiff_path) as image_dataset:
            crs_transformer = RasterioCRSTransformer(image_dataset)
            od_file = ObjectDetectionGeoJSONStore(labels_path, crs_transformer,
                                                  class_map)
            od_file.save(labels)
    elif task == 'semantic_segmentation':
        label_image = np.zeros((ymax, xmax, 1)).astype(np.uint8)

        for box, class_id in zip(boxes, class_ids):
            label_image[box.ymin:box.ymax, box.xmin:box.xmax, 0] = class_id

        # save labels to raster
        with rasterio.open(
                labels_path,
                'w',
                driver='GTiff',
                height=ymax,
                transform=transform,
                crs='EPSG:4326',
                compression=rasterio.enums.Compression.none,
                width=xmax,
                count=1,
                dtype='uint8') as dst:
            dst.write(label_image[:, :, 0], 1)


if __name__ == '__main__':
    generate_scene()
