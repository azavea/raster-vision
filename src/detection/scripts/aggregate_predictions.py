import json
import argparse
from os import makedirs
from os.path import join, splitext

import numpy as np
from scipy.ndimage import imread
import matplotlib as mpl
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
from cv2 import groupRectangles
import rasterio

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
        classes.extend([int(class_id) for class_id in preds['classes']])

    return boxes, scores, classes


def plot_predictions(plot_path, im, category_index, boxes, scores, classes):
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


# From https://stackoverflow.com/questions/28723670/intersection-over-union-between-two-detections # noqa
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def rect_to_bbox(rect):
    x, y, width, height = rect
    return [x, y, x + width, y + height]


def group_boxes(boxes, scores, im_size):
    '''Group boxes belonging to a single class.'''
    # Convert boxes to opencv rectangles
    rects = []
    for box_ind in range(boxes.shape[0]):
        box = boxes[box_ind, :].tolist()
        rect = box_to_cv2_rect(im_size, box)
        rects.append(rect)

    # Add last rect again to ensure that there are at least two rectangles
    # which seems to be required by groupRectangles.
    rects.append(rect)

    # Group the rects
    group_threshold = 1
    # May need to tune this parameter for other datasets depending on size
    # of detected objects.
    eps = 0.5
    grouped_rects = groupRectangles(rects, group_threshold, eps)[0]
    grouped_boxes = []
    grouped_scores = []

    # Find the rects and corresponding scores that best match the grouped_rects
    for grouped_rect in grouped_rects:
        bbox1 = rect_to_bbox(grouped_rect)
        best_iou = 0.0
        best_ind = None

        for rect_ind, rect in enumerate(rects[:-1]):
            bbox2 = rect_to_bbox(rect)
            iou = bb_intersection_over_union(bbox1, bbox2)
            if iou > best_iou:
                best_iou = iou
                best_ind = rect_ind

        grouped_boxes.append(cv2_rect_to_box(im_size, rects[best_ind]))
        grouped_scores.append(scores[best_ind])

    return grouped_boxes, grouped_scores


def group_predictions(boxes, classes, scores, im_size):
    '''For each class, group boxes that are overlapping.'''
    unique_classes = list(set(classes))

    boxes = np.array(boxes)
    classes = np.array(classes)
    scores = np.array(scores)

    grouped_boxes = []
    grouped_classes = []
    grouped_scores = []

    for class_id in unique_classes:
        class_boxes = boxes[classes == class_id]
        class_scores = scores[classes == class_id]

        class_grouped_boxes, class_grouped_scores = \
            group_boxes(class_boxes, class_scores, im_size)
        grouped_boxes.extend(class_grouped_boxes)
        grouped_classes.extend([class_id] * len(class_grouped_boxes))
        grouped_scores.extend(class_grouped_scores)

    return grouped_boxes, grouped_classes, grouped_scores


def save_geojson(path, boxes, classes, scores, im_size, category_index,
                 image_dataset=None):
    polygons = []
    for box in boxes:
        x, y, width, height = box_to_cv2_rect(im_size, box)
        nw = (x, y)
        ne = (x + width, y)
        se = (x + width, y + height)
        sw = (x, y + height)
        polygon = [nw, ne, se, sw, nw]
        # Transform from pixel coords to spatial coords
        if image_dataset:
            polygon = [image_dataset.ul(point[1], point[0])
                       for point in polygon]
        polygons.append(polygon)

    crs = None
    if image_dataset:
        # XXX not sure if I'm getting this properly
        crs_name = image_dataset.crs['init']
        crs = {
            'type': 'name',
            'properties': {
                'name': crs_name
            }
        }

    features = [{
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
        for polygon, class_id, score in zip(polygons, classes, scores)
    ]

    geojson = {
        'type': 'FeatureCollection',
        'crs': crs,
        'features': features
    }

    with open(path, 'w') as json_file:
        json.dump(geojson, json_file, indent=4)


def aggregate_predictions(image_path, window_info_path, predictions_path,
                          label_map_path, output_dir):
    print('Aggregating predictions over windows...')

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=37, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image_dataset = None
    if splitext(image_path)[1] == '.tif':
        image_dataset = rasterio.open(image_path)
        im = image_dataset.read()
        im = np.transpose(im, [1, 2, 0])
    else:
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
    boxes, classes, scores = group_predictions(boxes, classes, scores, im_size)

    agg_predictions_path = join(output_dir, 'agg_predictions.json')
    save_geojson(agg_predictions_path, boxes, classes, scores, im_size,
                 category_index, image_dataset=image_dataset)

    plot_path = join(output_dir, 'agg_predictions.jpg')
    plot_predictions(plot_path, im, category_index, boxes, scores, classes)


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
