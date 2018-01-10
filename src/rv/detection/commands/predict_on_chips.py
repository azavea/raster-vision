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

from object_detection.utils import (
    label_map_util, visualization_utils as vis_util)
from object_detection.utils.np_box_list_ops import (
    scale, filter_scores_greater_than)
from object_detection.utils.np_box_list import BoxList

from rv.detection.commands.settings import (
    max_num_classes, min_score_threshold, line_thickness)
from rv.utils.misc import save_img
from rv.utils.files import make_dir

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

    (boxes, scores, classes) = sess.run(
        [boxes, scores, classes],
        feed_dict={image_tensor: image_np_expanded})
    boxlist = BoxList(np.squeeze(boxes))
    boxlist.add_field('scores', np.squeeze(scores))
    boxlist.add_field('classes', np.squeeze(classes).astype(np.int32))

    return boxlist


def write_predictions_csv(predictions, predictions_path):
    with open(predictions_path, 'w') as predictions_file:
        json.dump(predictions, predictions_file)


def _predict_on_chips(inference_graph_path, label_map_path, chips_dir,
                      predictions_path, predictions_debug_dir=None):
    make_dir(predictions_path, use_dirname=True)
    if predictions_debug_dir is not None:
        make_dir(predictions_debug_dir)

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
                height, width = image_np.shape[0:2]

                norm_boxlist = compute_prediction(
                    image_np, detection_graph, sess)
                boxlist = scale(norm_boxlist, width, height)

                if predictions_debug_dir is not None:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        boxlist.get(),
                        boxlist.get_field('classes'),
                        boxlist.get_field('scores'),
                        category_index,
                        use_normalized_coordinates=False,
                        line_thickness=line_thickness)

                    debug_image_path = \
                        join(predictions_debug_dir, basename(image_path))
                    save_img(debug_image_path, image_np)

                filename = basename(image_path)
                filtered_boxlist = filter_scores_greater_than(
                    boxlist, min_score_threshold)
                predictions[filename] = {
                    'boxes': filtered_boxlist.get().tolist(),
                    'scores': filtered_boxlist.get_field('scores').tolist(),
                    'classes': filtered_boxlist.get_field('classes').tolist()
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
    """Make and save predictions over a directory of images."""
    _predict_on_chips(inference_graph_path, label_map_path, chips_dir,
                      predictions_path,
                      predictions_debug_dir=predictions_debug_dir)


if __name__ == '__main__':
    predict_on_chips()
