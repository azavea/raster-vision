import io
import numpy as np
import tensorflow as tf
import uuid

from os.path import join
from PIL import Image

from object_detection.utils import dataset_util
from rastervision.core.ml_backend import MLBackend
from rastervision.utils.files import make_dir


def numpy_to_png(array):
    im = Image.fromarray(array).convert('L')
    output = io.BytesIO()
    im.save(output, 'png')
    return output.getvalue()


def png_to_numpy(png):
    incoming = io.ByteesIO(png)
    im = Image.open(incoming)
    width, height = im.size
    return np.array(im.getdata(), dtype=np.uint8).reshape(height, width)


def write_tf_record(tf_examples, output_path):
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for tf_example in tf_examples:
            writer.write(tf_example.SerializeToString())


def make_tf_examples(training_data, class_map):
    tf_examples = []
    print('Creating TFRecord', end='', flush=True)
    for chip, window, labels in training_data:
        tf_example = create_tf_example(chip, window, labels, class_map)
        tf_examples.append(tf_example)
        print('.', end='', flush=True)
    print()
    return tf_examples


def create_tf_example(image, window, labels, class_map, chip_id=''):

    class_keys = set(class_map.get_keys())

    def fn(n):
        return (n if n in class_keys else 0)

    filtered_labels = np.vectorize(fn)(labels)
    filtered_labels = np.array(filtered_labels, dtype=np.uint8)

    image_encoded = numpy_to_png(image)
    image_filename = chip_id.encode('utf8')
    image_format = 'png'.encode('utf8')
    image_height, image_width, image_channels = image.shape
    image_segmentation_class_encoded = numpy_to_png(filtered_labels)
    image_segmentation_class_format = 'png'.encode('utf8')

    # import pdb ; pdb.set_trace()
    features = tf.train.Features(feature={
        'image/encoded': dataset_util.bytes_feature(image_encoded),
        'image/filename': dataset_util.bytes_feature(image_filename),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/channels': dataset_util.int64_feature(image_channels),
        'image/segmentation/class/encoded': dataset_util.bytes_feature(image_segmentation_class_encoded),
        'image/segmentation/class/format': dataset_util.bytes_feature(image_segmentation_class_format),
    })

    return tf.train.Example(features=features)


class TFDeeplab(MLBackend):

    def __init__(self):
        # persist scene training packages for when output_uri is remote
        self.scene_training_packages = []

    def process_scene_data(self, scene, data, class_map, options):
        base_uri = options.output_uri

        make_dir(base_uri)

        # import pdb ; pdb.set_trace()
        tf_examples = make_tf_examples(data, class_map)
        split = '{}-{}'.format(scene.id, uuid.uuid4())
        record_path = join(base_uri, '{}.record'.format(split))
        write_tf_record(tf_examples, record_path)

        return record_path

    def process_sceneset_results(self, training_results, validation_results,
                                 class_map, options):
        return 1

    def train(self, options):
        return 1

    def predict(self, chip, options):
        return 1
