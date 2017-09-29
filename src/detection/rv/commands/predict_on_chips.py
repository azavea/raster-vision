import numpy as np
from os import makedirs
from os.path import join, dirname, basename
import glob
import json

import click
import tensorflow as tf
# For headless environments
import matplotlib as mpl
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imsave

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from rv.commands.settings import (
    max_num_classes, min_score_threshold, line_thickness)

image_size = (12, 8)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_frozen_graph(inference_graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def compute_prediction(image_np, detection_graph, sess):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name(
        'image_tensor:0')
    boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return (boxes, scores, classes, num_detections)


def write_predictions_csv(predictions, predictions_path):
    with open(predictions_path, 'w') as predictions_file:
        json.dump(predictions, predictions_file)


def _predict_on_chips(inference_graph_path, label_map_path, chips_dir,
                      predictions_path, predictions_debug_dir=None):
    makedirs(dirname(predictions_path), exist_ok=True)
    if predictions_debug_dir is not None:
        makedirs(predictions_debug_dir, exist_ok=True)

    detection_graph = load_frozen_graph(inference_graph_path)

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=max_num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image_paths = glob.glob(join(chips_dir, '*.png'))
    predictions = {}

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_ind, image_path in enumerate(image_paths):
                click.echo('Computing predictions for {}'.format(image_path))
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)

                boxes, scores, classes, num_detections = \
                    compute_prediction(image_np, detection_graph, sess)

                if predictions_debug_dir is not None:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=line_thickness)

                    debug_image_path = \
                        join(predictions_debug_dir, basename(image_path))
                    imsave(debug_image_path, image_np)

                filename = basename(image_path)
                detections = scores > min_score_threshold
                predictions[filename] = {
                    'boxes': boxes[detections, :].tolist(),
                    'scores': scores[detections].tolist(),
                    'classes': classes[detections].tolist()
                }

    write_predictions_csv(predictions, predictions_path)


@click.command()
@click.argument('inference_graph_path')
@click.argument('label_map_path')
@click.argument('chips_dir')
@click.argument('predictions_path')
@click.option('--predictions-debug-dir', default=None)
def predict_on_chips(inference_graph_path, label_map_path, chips_dir,
                     predictions_path, predictions_debug_dir=None):
    """Make predictions over a directory of images and write results to CSV."""
    _predict_on_chips(inference_graph_path, label_map_path, chips_dir,
                      predictions_path,
                      predictions_debug_dir=predictions_debug_dir)


if __name__ == '__main__':
    predict_on_chips()
