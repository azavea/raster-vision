import json
import argparse
from os import makedirs
from os.path import join

import numpy as np
from scipy.ndimage import imread
import matplotlib as mpl
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def compute_agg_predictions(window_offsets, window_size, im_size, predictions):
    boxes = []
    scores = []
    classes = []

    y_im_size, x_im_size = im_size

    for file_name, preds in predictions.items():
        x, y = window_offsets[file_name]

        for box in preds['boxes']:
            # box is (ymin, xmin, ymax, xmax) in relative coords
            # (eg. 0.5 is middle of axis).
            # x, y are in pixel offsets.
            box = np.array(box) * window_size

            box[0] += y  # ymin
            box[1] += x  # xmin
            box[2] += y  # ymax
            box[3] += x  # xmax

            box[0] /= y_im_size
            box[1] /= x_im_size
            box[2] /= y_im_size
            box[3] /= x_im_size

            box = np.clip(box, 0, 1).tolist()
            boxes.append(box)

        scores.extend(preds['scores'])
        classes.extend(preds['classes'])

    return boxes, scores, classes


def plot_predictions(plot_path, im, label_map_path, boxes, scores, classes):
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=37, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    vis_util.visualize_boxes_and_labels_on_image_array(
        im,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    image_size = (24, 16)
    plt.figure(figsize=image_size)
    plt.imshow(im)
    plt.savefig(plot_path)


def aggregate_predictions(image_path, window_info_path, predictions_path,
                          label_map_path, output_dir):
    print('Aggregating predictions over windows...')
    im = imread(image_path)
    im_size = im.shape[0:2]

    with open(window_info_path) as window_info_file:
        window_info = json.load(window_info_file)
        window_offsets = window_info['offsets']
        window_size = window_info['window_size']

    with open(predictions_path) as predictions_file:
        predictions = json.load(predictions_file)

    makedirs(output_dir, exist_ok=True)
    boxes, scores, classes = compute_agg_predictions(
        window_offsets, window_size, im_size, predictions)

    agg_predictions = {
        'boxes': boxes,
        'scores': scores,
        'classes': classes
    }
    agg_predictions_path = join(output_dir, 'agg_predictions.json')
    with open(agg_predictions_path, 'w') as agg_predictions_file:
        json.dump(agg_predictions, agg_predictions_file)

    plot_path = join(output_dir, 'agg_predictions.jpg')
    plot_predictions(plot_path, im, label_map_path, boxes, scores, classes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path')
    parser.add_argument('--window-info-path')
    parser.add_argument('--predictions-path')
    parser.add_argument('--label-map-path')
    parser.add_argument('--output-dir')

    return parser.parse_args()


def run():
    args = parse_args()
    print('image_path: {}'.format(args.image_path))
    print('window_info_path: {}'.format(args.window_info_path))
    print('predictions_path: {}'.format(args.predictions_path))
    print('labels_map_path: {}'.format(args.label_map_path))
    print('output_dir: {}'.format(args.output_dir))

    aggregate_predictions(
        args.image_path, args.window_info_path, args.predictions_path,
        args.label_map_path, args.output_dir)


if __name__ == '__main__':
    run()
