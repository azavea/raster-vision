import json
from os import makedirs
from os.path import dirname

import click
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # NOQA
import rasterio
from pyproj import Proj, transform

from object_detection.utils import (
    label_map_util, visualization_utils as vis_util)
from object_detection.utils.np_box_list import BoxList
from object_detection.utils.np_box_list_ops import (
    clip_to_window, concatenate, multi_class_non_max_suppression)

from rv.commands.settings import (
    max_num_classes, line_thickness, planet_channel_order)
from rv.commands.utils import (
    load_window, translate_boxlist, save_img)


def compute_agg_predictions(chip_size, im_size, filename_to_chip_offset,
                            filename_to_boxlist):
    '''Aggregate chip predictions into predictions for original image.'''
    width, height = im_size
    boxlists = []
    filenames = sorted(filename_to_boxlist.keys())

    for filename in filenames:
        x_offset, y_offset = filename_to_chip_offset[filename]
        # box is in chip frame of reference
        file_boxlist = filename_to_boxlist[filename]
        if file_boxlist is not None:
            # translate boxes back to original image frame of reference
            file_boxlist = translate_boxlist(file_boxlist, x_offset, y_offset)

            # clip to edge of original image
            # normalize
            image_window = [0, 0, height, width]
            file_boxlist = clip_to_window(file_boxlist, image_window)
            boxlists.append(file_boxlist)

    if len(boxlists) == 0:
        boxlist = BoxList(np.zeros((0, 4)))
        boxlist.add_field('classes', np.zeros((0,)))
        boxlist.add_field('scores', np.zeros((0,)))
    else:
        boxlist = concatenate(boxlists)

    return boxlist


def plot_predictions(plot_path, im, category_index, boxlist):
    min_val = np.min(im)
    max_val = np.max(im)
    norm_im = 256 * ((im - min_val) / (max_val - min_val))
    norm_im = norm_im.astype(np.uint8)

    vis_util.visualize_boxes_and_labels_on_image_array(
        norm_im,
        boxlist.get(),
        boxlist.get_field('classes'),
        boxlist.get_field('scores'),
        category_index,
        use_normalized_coordinates=False,
        line_thickness=line_thickness,
        max_boxes_to_draw=None)

    save_img(plot_path, norm_im)


def save_geojson(path, boxlist, im_size, category_index, image_dataset=None):
    if image_dataset:
        src_crs = image_dataset.crs['init']
        src_proj = Proj(init=src_crs)
        # Convert to lat/lng
        dst_crs = 'epsg:4326'
        dst_proj = Proj(init=dst_crs)

    polygons = []
    for box in boxlist.get():
        ymin, xmin, ymax, xmax = box

        # four corners
        nw = (ymin, xmin)
        ne = (ymin, xmax)
        se = (ymax, xmax)
        sw = (ymax, xmin)
        polygon = [nw, ne, se, sw, nw]
        # Transform from pixel coords to spatial coords
        if image_dataset:
            dst_polygon = []
            for point in polygon:
                src_crs_point = image_dataset.ul(point[0], point[1])
                dst_crs_point = transform(
                    src_proj, dst_proj, src_crs_point[0], src_crs_point[1])
                dst_polygon.append(dst_crs_point)
        polygons.append(dst_polygon)

    crs = None
    if image_dataset:
        crs = {
            'type': 'name',
            'properties': {
                'name': dst_crs
            }
        }

    features = []
    classes = boxlist.get_field('classes')
    scores = boxlist.get_field('scores')

    for polygon, class_id, score in zip(polygons, classes, scores):
        feature = {
            'type': 'Feature',
            'properties': {
                'class_id': int(class_id),
                'class_name': category_index[class_id]['name'],
                'score': score

            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            }
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'crs': crs,
        'features': features
    }

    with open(path, 'w') as json_file:
        json.dump(geojson, json_file, indent=4)


def load_predictions(predictions_path):
    filename_to_boxlist = {}
    with open(predictions_path) as predictions_file:
        predictions = json.load(predictions_file)

        for filename, prediction in predictions.items():
            boxes = np.array(prediction['boxes'], dtype=float)
            classes = np.array(prediction['classes'], dtype=int)
            scores = np.array(prediction['scores'], dtype=float)
            if boxes.shape[0] > 0:
                boxlist = BoxList(boxes)
                boxlist.add_field('classes', classes)
                boxlist.add_field('scores', scores)
                filename_to_boxlist[filename] = boxlist
            else:
                filename_to_boxlist[filename] = None

    return filename_to_boxlist


def _aggregate_predictions(image_path, chip_info_path, predictions_path,
                           label_map_path, agg_predictions_path,
                           agg_predictions_debug_path=None,
                           channel_order=planet_channel_order,
                           score_thresh=0.5, merge_thresh=0.05):
    click.echo('Aggregating predictions over chips...')

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=max_num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image_dataset = rasterio.open(image_path)
    im_size = [image_dataset.width, image_dataset.height]

    with open(chip_info_path) as chip_info_file:
        chip_info = json.load(chip_info_file)
        filename_to_chip_offset = chip_info['offsets']
        chip_size = chip_info['chip_size']

    filename_to_boxlist = load_predictions(predictions_path)

    boxlist = compute_agg_predictions(
        chip_size, im_size, filename_to_chip_offset, filename_to_boxlist)

    # Due to the sliding window approach, sometimes there are multiple
    # slightly different detections where there should only be one. So
    # we group them together.
    max_output_size = 10000
    boxlist = multi_class_non_max_suppression(
        boxlist, score_thresh, merge_thresh, max_output_size)

    # I'm not sure if this is a bug in TF Object Detection API or a
    # misunderstanding on my part, but the multi_class_non_max_suppression
    # function returns class ids >= 0, but the label map enforces a
    # constraint that ids are > 0. So I just add one to the output to get
    # back to ids that are > 0.
    classes = boxlist.get_field('classes')
    classes += 1

    makedirs(dirname(agg_predictions_path), exist_ok=True)
    save_geojson(agg_predictions_path, boxlist, im_size,
                 category_index, image_dataset=image_dataset)

    if agg_predictions_debug_path is not None:
        makedirs(dirname(agg_predictions_debug_path), exist_ok=True)
        im = load_window(image_dataset, channel_order)
        plot_predictions(
            agg_predictions_debug_path, im, category_index, boxlist)


@click.command()
@click.argument('image_path')
@click.argument('chip_info_path')
@click.argument('predictions_path')
@click.argument('label_map_path')
@click.argument('agg_predictions_path')
@click.option('--agg-predictions-debug-path', default=None,
              help='Path to aggregate predictions debug plot')
@click.option('--channel-order', nargs=3, type=int,
              default=planet_channel_order, help='Indices of RGB channels')
@click.option('--score-thresh', default=0.5,
              help='Score threshold of predictions to keep')
@click.option('--merge-thresh', default=0.05,
              help='IOU threshold for merging predictions')
def aggregate_predictions(image_path, chip_info_path, predictions_path,
                          label_map_path, agg_predictions_path,
                          agg_predictions_debug_path, channel_order,
                          score_thresh, merge_thresh):
    """Aggregate predictions from chips.

    Args:
        image_path: Original image that detection is being run over
        chip_info_path: Chip info file with coordinates of chips
        predictions_path: Path to predictions for each chip
        agg_prediction_path: Path to output GeoJSON file with aggregate
            predictions
    """
    _aggregate_predictions(image_path, chip_info_path, predictions_path,
                           label_map_path, agg_predictions_path,
                           agg_predictions_debug_path=agg_predictions_debug_path,  # noqa
                           channel_order=channel_order,
                           score_thresh=score_thresh,
                           merge_thresh=merge_thresh)


if __name__ == '__main__':
    aggregate_predictions()
