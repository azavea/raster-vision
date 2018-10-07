import unittest
from google.protobuf import json_format
import json

import rastervision as rv
from rastervision.core.class_map import ClassItem
from rastervision.protos.task_pb2 import TaskConfig as TaskConfigMsg
from rastervision.protos.class_item_pb2 import ClassItem as ClassItemMsg


class TestSemanticSegmentationConfig(unittest.TestCase):
    def test_build_task(self):
        classes = ['one', 'two']
        expected = [ClassItem(1, 'one'), ClassItem(2, 'two')]

        t = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                         .with_classes(classes) \
                         .with_chip_options(target_classes=[1, 2]) \
                         .build()

        self.assertEqual(t.task_type, rv.SEMANTIC_SEGMENTATION)
        self.assertListEqual(t.class_map.get_items(), expected)
        self.assertListEqual(t.chip_options.target_classes, [1, 2])

    def test_build_task_from_proto(self):
        task_config = {
            'task_type': rv.SEMANTIC_SEGMENTATION,
            'semantic_segmentation_config': {
                'chip_options': {
                    'debug_chip_probability': 0.75
                },
                'chip_size':
                500,
                'class_items': [{
                    'id': 1,
                    'name': 'car',
                    'color': 'red'
                }, {
                    'id': 2,
                    'name': 'building',
                    'color': 'blue'
                }]
            }
        }
        msg = json_format.Parse(json.dumps(task_config), TaskConfigMsg())
        task = rv.TaskConfig.from_proto(msg)

        self.assertEqual(task.class_map.get_by_name('building').id, 2)
        self.assertEqual(task.chip_size, 500)
        self.assertEqual(task.chip_options.debug_chip_probability, 0.75)

    def test_create_proto_from_task(self):
        t = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                         .with_classes(['car', 'boat']) \
                         .with_chip_size(500) \
                         .build()

        msg = t.to_proto()

        expected_classes = [
            ClassItemMsg(name='car', id=1),
            ClassItemMsg(name='boat', id=2)
        ]

        self.assertEqual(msg.task_type, rv.SEMANTIC_SEGMENTATION)
        self.assertEqual(msg.semantic_segmentation_config.chip_size, 500)

        actual_class_items = dict(
            [(i.id, i) for i in msg.semantic_segmentation_config.class_items])
        expected_class_items = dict([(i.id, i) for i in expected_classes])

        self.assertDictEqual(actual_class_items, expected_class_items)


if __name__ == '__main__':
    unittest.main()
