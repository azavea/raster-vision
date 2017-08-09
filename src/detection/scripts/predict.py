import numpy as np
import os
import glob
import json

import tensorflow as tf
# For headless environments
import matplotlib as mpl
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


flags = tf.app.flags
flags.DEFINE_string('frozen_graph_path', '', 'Path to frozen inference graph.')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_string('input_dir', '', 'Path to directory of input images.')
flags.DEFINE_string('output_dir', '', 'Path to output images')
FLAGS = flags.FLAGS

image_size = (12, 8)
min_score_threshold = 0.5


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_frozen_graph(frozen_graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
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


def main(_):
    output_images_dir = os.path.join(FLAGS.output_dir, 'images')
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)

    detection_graph = load_frozen_graph(FLAGS.frozen_graph_path)

    label_map = label_map_util.load_labelmap(FLAGS.label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=37, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image_paths = glob.glob(os.path.join(FLAGS.input_dir, '*.jpg'))
    predictions = {}
    predictions_path = os.path.join(
        FLAGS.output_dir, 'predictions.csv')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_ind, image_path in enumerate(image_paths):
                print('Computing predictions for {}'.format(image_path))
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)

                boxes, scores, classes, num_detections = \
                    compute_prediction(image_np, detection_graph, sess)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                plt.figure(figsize=image_size)
                plt.imshow(image_np)
                out_image_path = os.path.join(
                    output_images_dir, os.path.basename(image_path))
                plt.savefig(out_image_path)

                filename = os.path.basename(image_path)
                detections = scores > min_score_threshold
                predictions[filename] = {
                    'boxes': boxes[detections, :].tolist(),
                    'scores': scores[detections].tolist(),
                    'classes': classes[detections].tolist()
                }

    write_predictions_csv(predictions, predictions_path)


if __name__ == '__main__':
    tf.app.run()
