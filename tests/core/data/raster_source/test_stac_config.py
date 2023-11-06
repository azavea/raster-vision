import unittest

from pystac import Item, ItemCollection

from rastervision.core.data.raster_source import (STACItemConfig,
                                                  STACItemCollectionConfig)

from tests import data_file_path


class TestSTACItemConfig(unittest.TestCase):
    def test_build(self):
        uri = data_file_path('stac/item.json')
        cfg = STACItemConfig(uri=uri, assets=['red'])
        item = cfg.build()
        self.assertIsInstance(item, Item)
        self.assertEqual(len(item.assets), 1)
        self.assertIn('red', item.assets)


class TestSTACItemCollectionConfig(unittest.TestCase):
    def test_build(self):
        uri = data_file_path('stac/item_collection.json')
        cfg = STACItemCollectionConfig(uri=uri, assets=['red'])
        items = cfg.build()
        self.assertIsInstance(items, ItemCollection)
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertEqual(len(item.assets), 1)
            self.assertIn('red', item.assets)


if __name__ == '__main__':
    unittest.main()
