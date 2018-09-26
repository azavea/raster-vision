from copy import deepcopy
from typing import (List, Dict, Tuple, Union)

import rastervision as rv
from rastervision.task import ChipClassification
from rastervision.core.class_map import (ClassMap, ClassItem)
from rastervision.task import (TaskConfig, TaskConfigBuilder)
from rastervision.protos.task_pb2 import TaskConfig as TaskConfigMsg
from rastervision.protos.class_item_pb2 import ClassItem as ClassItemMsg


class ChipClassificationConfig(TaskConfig):
    def __init__(self,
                 class_map,
                 predict_batch_size=10,
                 predict_package_uri=None,
                 debug=False,
                 predict_debug_uri=None,
                 chip_size=300):
        super().__init__(rv.CHIP_CLASSIFICATION, predict_batch_size,
                         predict_package_uri, debug, predict_debug_uri)
        self.class_map = class_map
        self.chip_size = chip_size

    def create_task(self, backend):
        return ChipClassification(self, backend)

    def to_proto(self):
        conf = TaskConfigMsg.ChipClassificationConfig(
            chip_size=self.chip_size, class_items=self.class_map.to_proto())
        return TaskConfigMsg(
            task_type=rv.CHIP_CLASSIFICATION, chip_classification_config=conf)

    def save_bundle_files(self, bundle_dir):
        return (self, [])

    def load_bundle_files(self, bundle_dir):
        return self


class ChipClassificationConfigBuilder(TaskConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'class_map': prev.class_map,
                'chip_size': prev.chip_size,
                'predict_batch_size': prev.predict_batch_size,
                'predict_package_uri': prev.predict_package_uri,
                'debug': prev.debug,
                'predict_debug_uri': prev.predict_debug_uri
            }
        super().__init__(ChipClassificationConfig, config)

    def validate(self):
        if 'class_map' not in self.config:
            raise rv.ConfigError('Class map required for this task. '
                                 'Use "with_classes"')

    def from_proto(self, msg):
        b = super().from_proto(msg)
        conf = msg.chip_classification_config
        return b.with_classes(list(conf.class_items)) \
                .with_chip_size(conf.chip_size)

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
