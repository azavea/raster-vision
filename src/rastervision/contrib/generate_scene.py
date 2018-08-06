import random

import click
import numpy as np
import rasterio
from rasterio.transform import from_origin

from rastervision.core.box import Box
from rastervision.crs_transformers.rasterio_crs_transformer import (
    RasterioCRSTransformer)
from rastervision.labels.object_detection_labels import ObjectDetectionLabels
from rastervision.label_stores.object_detection_geojson_file import (
    ObjectDetectionGeoJSONFile)
from rastervision.core.class_map import ClassItem, ClassMap


@click.command()
@click.argument('tiff_path')
@click.argument('labels_path')
def generate_scene(tiff_path, labels_path):
    """Generate a synthetic object detection scene.

    Randomly generates a GeoTIFF with red and greed boxes denoting two
    classes and a corresponding label file. This is useful for generating
    synthetic scenes for testing purposes.
    """
    class_map = ClassMap([ClassItem(1, 'car'), ClassItem(2, 'building')])

    # make extent that's divisible by chip_size
    chip_size = 300
    y_len = 2
    x_len = 2
    ymax = y_len * chip_size
    xmax = x_len * chip_size
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
            box = window.make_random_square(50).as_int()

            boxes.append(box)
            class_ids.append(class_id)

            image[box.ymin:box.ymax, box.xmin:box.xmax, class_id - 1] = 255

    # save image as geotiff centered in philly
    transform = from_origin(-75.163506, 39.952536, 0.000001, 0.000001)

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

    # make an OD labels and make boxes
    npboxes = Box.to_npboxes(boxes)
    class_ids = np.array(class_ids)
    labels = ObjectDetectionLabels(npboxes, class_ids)

    # save labels to geojson
    image_dataset = rasterio.open(tiff_path)
    crs_transformer = RasterioCRSTransformer(image_dataset)
    od_file = ObjectDetectionGeoJSONFile(
        labels_path, crs_transformer, class_map, readable=False, writable=True)
    od_file.set_labels(labels)
    od_file.save()


if __name__ == '__main__':
    generate_scene()
