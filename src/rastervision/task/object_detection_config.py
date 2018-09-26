from copy import deepcopy
from typing import (List, Dict, Tuple, Union)

import rastervision as rv
from rastervision.task import ObjectDetection
from rastervision.core.class_map import (ClassMap, ClassItem)
from rastervision.task import (TaskConfig, TaskConfigBuilder)
from rastervision.protos.task_pb2 import TaskConfig as TaskConfigMsg
from rastervision.protos.class_item_pb2 import ClassItem as ClassItemMsg


class ObjectDetectionConfig(TaskConfig):
    class ChipOptions:
        def __init__(self,
                     neg_ratio=1,
                     ioa_thresh=0.8,
                     window_method='chip',
                     label_buffer=0.0):
            self.neg_ratio = neg_ratio
            self.ioa_thresh = ioa_thresh
            self.window_method = window_method
            self.label_buffer = label_buffer

    class PredictOptions:
        def __init__(self, merge_thresh=0.5, score_thresh=0.5):
            self.merge_thresh = merge_thresh
            self.score_thresh = score_thresh

    def __init__(self,
                 class_map,
                 predict_batch_size=10,
                 predict_package_uri=None,
                 debug=True,
                 predict_debug_uri=None,
                 chip_size=300,
                 chip_options=ChipOptions(),
                 predict_options=PredictOptions()):
        super().__init__(rv.OBJECT_DETECTION, predict_batch_size,
                         predict_package_uri, debug, predict_debug_uri)
        self.class_map = class_map
        self.chip_size = chip_size
        self.chip_options = chip_options
        self.predict_options = predict_options

    def create_task(self, backend):
        return ObjectDetection(self, backend)

    def to_proto(self):
        msg = super().to_proto()
        chip_options = TaskConfigMsg.ObjectDetectionConfig.ChipOptions(
            neg_ratio=self.chip_options.neg_ratio,
            ioa_thresh=self.chip_options.ioa_thresh,
            window_method=self.chip_options.window_method,
            label_buffer=self.chip_options.label_buffer)

        predict_options = TaskConfigMsg.ObjectDetectionConfig.PredictOptions(
            merge_thresh=self.predict_options.merge_thresh,
            score_thresh=self.predict_options.score_thresh)

        conf = TaskConfigMsg.ObjectDetectionConfig(
            chip_size=self.chip_size,
            class_items=self.class_map.to_proto(),
            chip_options=chip_options,
            predict_options=predict_options)
        msg.MergeFrom(TaskConfigMsg(object_detection_config=conf))

        return msg

    def save_bundle_files(self, bundle_dir):
        return (self, [])

    def load_bundle_files(self, bundle_dir):
        return self


class ObjectDetectionConfigBuilder(TaskConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'predict_batch_size': prev.predict_batch_size,
                'predict_package_uri': prev.predict_package_uri,
                'debug': prev.debug,
                'predict_debug_uri': prev.predict_debug_uri,
                'class_map': prev.class_map,
                'chip_size': prev.chip_size,
                'chip_options': prev.chip_options,
                'predict_options': prev.predict_options
            }
        super().__init__(ObjectDetectionConfig, config)

    def validate(self):
        if 'class_map' not in self.config:
            raise rv.ConfigError('Class map required for this task. '
                                 'Use "with_classes"')

    def from_proto(self, msg):
        b = super().from_proto(msg)
        conf = msg.object_detection_config

        return b.with_classes(list(conf.class_items)) \
                .with_chip_size(conf.chip_size) \
                .with_chip_options(neg_ratio=conf.chip_options.neg_ratio,
                                   ioa_thresh=conf.chip_options.ioa_thresh,
                                   window_method=conf.chip_options.window_method,
                                   label_buffer=conf.chip_options.label_buffer) \
                .with_predict_options(merge_thresh=conf.predict_options.merge_thresh,
                                      score_thresh=conf.predict_options.score_thresh)

    def with_classes(
            self, classes: Union[ClassMap, List[str], List[ClassItemMsg], List[
                ClassItem], Dict[str, int], Dict[str, Tuple[int, str]]]):
        """Set the classes for this task.

            Args:
                classes: Either a list of class names, a dict which
                         maps class names to class ids, or a dict
                         which maps class names to a tuple of (class_id, color),
                         where color is a PIL color string.
        """
        b = deepcopy(self)
        b.config['class_map'] = ClassMap.construct_from(classes)
        return b

    def with_chip_size(self, chip_size):
        """Set the chip_size for this task.

            Args:
                chip_size: Integer value chip size
        """
        b = deepcopy(self)
        b.config['chip_size'] = chip_size
        return b

    def with_chip_options(self,
                          neg_ratio=1,
                          ioa_thresh=0.8,
                          window_method='chip',
                          label_buffer=0.0):
        """Sets object detection configurations for the Chip command

        Args:
           neg_ratio: The ratio of negative chips (those containing no bounding boxes)
                      to positive chips. This can be useful if the statistics of the
                      background is different in positive chips. For example, in car
                      detection, the positive chips will always contain roads, but no
                      examples of rooftops since cars tend to not be near rooftops.
                      This option is not used when window_method is `sliding`.

           ioa_thresh: When a box is partially outside of a training chip, it is not
                       clear if (a clipped version) of the box should be included in
                       the chip. If the IOA (intersection over area) of the box with
                       the chip is greater than ioa_thresh, it is included in the chip.

           window_method: Different models in the Object Detection API have different
                          inputs. Some models allow variable size inputs so several
                          methods of building training data are required

                          Valid values are:
                            - chip (default)
                            - label
                               - each label's bounding box is the positive window
                            - image
                               - each image is the positive window
                            - sliding
                               - each image is from a sliding window with 50% overlap

            label_buffer: If method is "label", the positive window can be buffered.
                          If value is >= 0. and < 1., the value is treated as a percentage
                          If value is >= 1., the value is treated in number of pixels
        """
        b = deepcopy(self)
        b.config['chip_options'] = ObjectDetectionConfig.ChipOptions(
            neg_ratio=neg_ratio,
            ioa_thresh=ioa_thresh,
            window_method=window_method,
            label_buffer=label_buffer)
        return b

    def with_predict_options(self, merge_thresh=0.5, score_thresh=0.5):
        """Prediction options for this task.

        Args:
           merge_thresh: If predicted boxes have an IOA (intersection over area)
                         greater than merge_thresh, then they are merged into a
                         single box during postprocessing. This is needed since
                         the sliding window approach results in some false duplicates.

           score_thresh: Predicted boxes are only output if their
                         score is above score_thresh.
        """
        b = deepcopy(self)
        b.config['predict_options'] = ObjectDetectionConfig.PredictOptions(
            merge_thresh=merge_thresh, score_thresh=score_thresh)
        return b
