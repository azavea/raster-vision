import unittest
from google.protobuf import json_format
import json

import rastervision as rv
from rastervision.core.class_map import ClassItem
from rastervision.protos.task_pb2 import TaskConfig as TaskConfigMsg
from rastervision.protos.class_item_pb2 import ClassItem as ClassItemMsg


class TestChipClassificationConfig(unittest.TestCase):
    def test_build_task(self):
        classes = ['one', 'two']
        expected = [ClassItem(1, 'one'), ClassItem(2, 'two')]

        t = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                   .with_classes(classes) \
                   .build()

        self.assertEqual(t.task_type, rv.CHIP_CLASSIFICATION)
        self.assertListEqual(t.class_map.get_items(), expected)

    def test_build_task_from_proto(self):
        task_config = {
            'task_type': rv.CHIP_CLASSIFICATION,
            'chip_classification_config': {
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
                }, {
                    'id': 3,
                    'name': 'background',
                    'color': 'black'
                }]
            }
        }
        msg = json_format.Parse(json.dumps(task_config), TaskConfigMsg())

        task = rv.TaskConfig.from_proto(msg)

        self.assertEqual(task.class_map.get_by_name('building').id, 2)
        self.assertEqual(task.chip_size, 500)

    def test_create_proto_from_task(self):
        t = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                         .with_classes(['car', 'boat']) \
                         .with_chip_size(500) \
                         .build()

        msg = t.to_proto()

        expected_classes = [
            ClassItemMsg(name='car', id=1),
            ClassItemMsg(name='boat', id=2)
        ]

        self.assertEqual(msg.task_type, rv.CHIP_CLASSIFICATION)
        self.assertEqual(msg.chip_classification_config.chip_size, 500)

        actual_class_items = dict(
            [(i.id, i) for i in msg.chip_classification_config.class_items])
        expected_class_items = dict([(i.id, i) for i in expected_classes])

        self.assertDictEqual(actual_class_items, expected_class_items)

    def test_missing_config_class_map(self):
        with self.assertRaises(rv.ConfigError):
            rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION).build()

    def test_no_missing_config(self):
        try:
            rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION).with_classes(
                ['car']).build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')


if __name__ == '__main__':
    unittest.main()
