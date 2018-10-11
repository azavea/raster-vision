import unittest

from rastervision.core import (ClassMap, ClassItem)
from rastervision.protos.class_item_pb2 \
    import ClassItem as ClassItemMsg


class TestClassMap(unittest.TestCase):
    def test_construct_from_list_str(self):
        source = ['one', 'two', 'three']
        cm = ClassMap.construct_from(source)
        for i, name in enumerate(source):
            expected = ClassItem(id=i + 1, name=name, color=None)
            actual = cm.get_by_id(i + 1)
            self.assertEqual(actual, expected)

    def test_construct_from(self):
        self.assertRaises(Exception,
                          lambda: ClassMap.construct_from('some string'))

    def test_all_color_true(self):
        source = [
            ClassItemMsg(id=1, name='one', color='red'),
            ClassItemMsg(id=2, name='two', color='green'),
            ClassItemMsg(id=3, name='three', color='blue')
        ]
        cm = ClassMap.construct_from(source)
        self.assertTrue(cm.has_all_colors())

    def test_all_color_false(self):
        source = [
            ClassItemMsg(id=1, name='one', color='red'),
            ClassItemMsg(id=2, name='two', color='green'),
            ClassItemMsg(id=3, name='three')
        ]
        cm = ClassMap.construct_from(source)
        self.assertFalse(cm.has_all_colors())

    def test_category_index(self):
        source = [
            ClassItemMsg(id=1, name='one', color='red'),
            ClassItemMsg(id=2, name='two', color='green'),
            ClassItemMsg(id=3, name='three')
        ]
        cm = ClassMap.construct_from(source)
        index = cm.get_category_index()
        self.assertEqual(index[1], {'id': 1, 'name': 'one'})

    def test_get_by_name_negative(self):
        source = [
            ClassItemMsg(id=1, name='one', color='red'),
            ClassItemMsg(id=2, name='two', color='green'),
            ClassItemMsg(id=3, name='three', color='blue')
        ]
        cm = ClassMap.construct_from(source)
        self.assertRaises(ValueError, lambda: cm.get_by_name('four'))

    def test_construct_from_protos(self):
        source = [
            ClassItemMsg(id=1, name='one', color='red'),
            ClassItemMsg(id=2, name='two', color='green'),
            ClassItemMsg(id=3, name='three', color='blue')
        ]
        cm = ClassMap.construct_from(source)
        for i, msg in enumerate(source):
            expected = ClassItem.from_proto(msg)
            actual = cm.get_by_id(i + 1)
            self.assertEqual(actual, expected)

    def test_construct_from_class_items(self):
        source = [
            ClassItem(id=1, name='one', color='red'),
            ClassItem(id=2, name='two', color='green'),
            ClassItem(id=3, name='three', color='blue')
        ]
        cm = ClassMap.construct_from(source)
        for i, item in enumerate(source):
            expected = item
            actual = cm.get_by_id(i + 1)
            self.assertEqual(actual, expected)

    def test_construct_from_dict_no_color(self):
        source = {'one': 1, 'two': 2, 'three': 3}
        cm = ClassMap.construct_from(source)
        for name in source:
            expected = ClassItem(id=source[name], name=name, color=None)
            actual = cm.get_by_id(source[name])
            self.assertEqual(actual, expected)

    def test_construct_from_dict_with_color(self):
        source = {'one': (1, 'red'), 'two': (2, 'green'), 'three': (3, 'blue')}
        cm = ClassMap.construct_from(source)
        for name in source:
            expected = ClassItem(
                id=source[name][0], name=name, color=source[name][1])
            actual = cm.get_by_id(source[name][0])
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
