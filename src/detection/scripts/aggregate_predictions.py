import json
import argparse
from os import makedirs
from os.path import join

import numpy as np
from scipy.ndimage import imread
import matplotlib as mpl
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
from cv2 import groupRectangles

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def compute_agg_predictions(window_offsets, window_size, im_size, predictions):
    ''' Aggregate window predictions into predictions for original image. '''
    boxes = []
    scores = []
    classes = []

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

            box[0] /= im_size[1]
            box[1] /= im_size[0]
            box[2] /= im_size[1]
            box[3] /= im_size[0]

            box = np.clip(box, 0, 1).tolist()
            boxes.append(box)

        scores.extend(preds['scores'])
        classes.extend([int(class_code) for class_code in preds['classes']])

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


def box_to_cv2_rect(im_size, box):
    ymin, xmin, ymax, xmax = box
    width = xmax - xmin
    height = ymax - ymin

    xmin = int(xmin * im_size[0])
    width = int(width * im_size[0])
    ymin = int(ymin * im_size[1])
    height = int(height * im_size[1])

    rect = (xmin, ymin, width, height)
    return rect


def cv2_rect_to_box(im_size, rect):
    x, y, width, height = rect

    x /= im_size[0]
    width /= im_size[0]
    y /= im_size[1]
    height /= im_size[1]

    box = [y, x, y + height, x + width]
    return box


def group_predictions(boxes, classes, im_size):
    '''For each class, group boxes that are overlapping.'''
    unique_classes = list(set(classes))

    boxes = np.array(boxes)
    classes = np.array(classes)

    grouped_boxes = []
    grouped_classes = []

    for class_code in unique_classes:
        class_boxes = boxes[classes == class_code]
        # Convert boxes to opencv rectangles
        nboxes = class_boxes.shape[0]
        rects = []
        for box_ind in range(nboxes):
            box = class_boxes[box_ind, :].tolist()
            rect = box_to_cv2_rect(im_size, box)

            # Convert to pixel offsets
            rects.append(rect)

        # Add last rect again to ensure that there are at least two rectangles
        # which seems to be required by groupRectangles.
        rects.append(rect)

        group_threshold = 1
        # May need to tune this parameter for other datasets depending on size
        # of detected objects.
        eps = 0.5
        grouped_rects, weights = groupRectangles(rects, group_threshold, eps)

        grouped_boxes.extend(
            [cv2_rect_to_box(im_size, rect) for rect in grouped_rects])
        grouped_classes.extend([class_code] * grouped_rects.shape[0])

    return grouped_boxes, grouped_classes


def aggregate_predictions(image_path, window_info_path, predictions_path,
                          label_map_path, output_dir):
    print('Aggregating predictions over windows...')
    im = imread(image_path)
    im_size = [im.shape[1], im.shape[0]]

    with open(window_info_path) as window_info_file:
        window_info = json.load(window_info_file)
        window_offsets = window_info['offsets']
        window_size = window_info['window_size']

    with open(predictions_path) as predictions_file:
        predictions = json.load(predictions_file)

    makedirs(output_dir, exist_ok=True)
    boxes, scores, classes = compute_agg_predictions(
        window_offsets, window_size, im_size, predictions)
    # Due to the sliding window approach, sometimes there are multiple
    # slightly different detections where there should only be one. So
    # we group them together.
    boxes, classes = group_predictions(boxes, classes, im_size)

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
