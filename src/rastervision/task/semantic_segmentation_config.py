from copy import deepcopy
from typing import (List, Dict, Tuple, Union)

import rastervision as rv
from rastervision.task import SemanticSegmentation
from rastervision.core.class_map import (ClassMap, ClassItem)
from rastervision.task import (TaskConfig, TaskConfigBuilder)
from rastervision.protos.task_pb2 import TaskConfig as TaskConfigMsg
from rastervision.protos.class_item_pb2 import ClassItem as ClassItemMsg


class SemanticSegmentationConfig(TaskConfig):
    class ChipOptions:
        def __init__(self,
                     window_method='random_sample',
                     target_classes=None,
                     debug_chip_probability=0.25,
                     negative_survival_probability=1.0,
                     chips_per_scene=1000,
                     target_count_threshold=1000,
                     stride=None):
            self.window_method = window_method
            self.target_classes = target_classes
            self.debug_chip_probability = debug_chip_probability
            self.negative_survival_probability = negative_survival_probability
            self.chips_per_scene = chips_per_scene
            self.target_count_threshold = target_count_threshold
            self.stride = stride

    def __init__(self,
                 class_map,
                 predict_batch_size=10,
                 predict_package_uri=None,
                 debug=True,
                 chip_size=300,
                 chip_options=None):
        super().__init__(rv.SEMANTIC_SEGMENTATION, predict_batch_size,
                         predict_package_uri, debug)
        self.class_map = class_map
        self.chip_size = chip_size
        if chip_options is None:
            chip_options = SemanticSegmentationConfig.ChipOptions()
        self.chip_options = chip_options

    def save_bundle_files(self, bundle_dir):
        return (self, [])

    def load_bundle_files(self, bundle_dir):
        return self

    def create_task(self, backend):
        return SemanticSegmentation(self, backend)

    def to_proto(self):
        msg = super().to_proto()
        chip_options = TaskConfigMsg.SemanticSegmentationConfig.ChipOptions(
            window_method=self.chip_options.window_method,
            target_classes=self.chip_options.target_classes,
            debug_chip_probability=self.chip_options.debug_chip_probability,
            negative_survival_probability=self.chip_options.
            negative_survival_probability,
            chips_per_scene=self.chip_options.chips_per_scene,
            target_count_threshold=self.chip_options.target_count_threshold,
            stride=self.chip_options.stride)

        conf = TaskConfigMsg.SemanticSegmentationConfig(
            chip_size=self.chip_size,
            class_items=self.class_map.to_proto(),
            chip_options=chip_options)
        msg.MergeFrom(TaskConfigMsg(semantic_segmentation_config=conf))

        return msg


class SemanticSegmentationConfigBuilder(TaskConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'predict_batch_size': prev.predict_batch_size,
                'predict_package_uri': prev.predict_package_uri,
                'debug': prev.debug,
                'class_map': prev.class_map,
                'chip_size': prev.chip_size,
                'chip_options': prev.chip_options
            }
        super().__init__(SemanticSegmentationConfig, config)

    def from_proto(self, msg):
        conf = msg.semantic_segmentation_config
        b = SemanticSegmentationConfigBuilder()

        negative_survival_probability = conf.chip_options \
                                            .negative_survival_probability

        return b.with_classes(list(conf.class_items)) \
                .with_predict_batch_size(msg.predict_batch_size) \
                .with_predict_package_uri(msg.predict_package_uri) \
                .with_debug(msg.debug) \
                .with_chip_size(conf.chip_size) \
                .with_chip_options(
                    window_method=conf.chip_options.window_method,
                    target_classes=conf.chip_options.target_classes,
                    debug_chip_probability=conf.chip_options.debug_chip_probability,
                    negative_survival_probability=negative_survival_probability,
                    chips_per_scene=conf.chip_options.chips_per_scene,
                    target_count_threshold=conf.chip_options.target_count_threshold,
                    stride=conf.chip_options.stride)

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
                          window_method='random_sample',
                          target_classes=None,
                          debug_chip_probability=0.25,
                          negative_survival_probability=1.0,
                          chips_per_scene=1000,
                          target_count_threshold=1000,
                          stride=None):
        """Sets semantic segmentation configurations for the Chip command

           Args:
            window_method: Window method to use for chipping.
                           Options are: random_sample, sliding
            target_classes: list of class ids to train model on
            debug_chip_probability: probability of generating a debug chip.
                                    Applies to the 'random_sample' window method.
            negative_survival_probability: probability that a sampled negative
                                           chip will be utilized if it does not
                                           contain more pixels than
                                           target_count_threshold.
                                           Applies to the 'random_sample' window method.
            chips_per_scene: number of chips to generate per scene.
                             Applies to the 'random_sample' window method.
            target_count_threshold: minimum number of pixels covering target_classes
                                    that a chip must have.
                                    Applies to the 'random_sample' window method.
            stride: Stride of windows across image. Defaults to half the chip size.
                    Applies to the 'sliding_window' method.

        Returns:
            SemanticSegmentationConfigBuilder
        """
        b = deepcopy(self)

        b.config['chip_options'] = SemanticSegmentationConfig.ChipOptions(
            window_method=window_method,
            target_classes=target_classes,
            debug_chip_probability=debug_chip_probability,
            negative_survival_probability=negative_survival_probability,
            chips_per_scene=chips_per_scene,
            target_count_threshold=target_count_threshold,
            stride=stride)
        return b
