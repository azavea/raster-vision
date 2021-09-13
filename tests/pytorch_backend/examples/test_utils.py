import unittest

from shapely.geometry import Polygon
from rastervision.pytorch_backend.examples.utils import read_stac

from tests import data_file_path


class TestUtils(unittest.TestCase):
    def test_read_stac(self):
        expected_keys = {
            'label_uri': str,
            'image_uris': list,
            'label_bbox': Polygon,
            'image_bbox': (type(None), Polygon),
            'bboxes_intersect': bool,
            'aoi_geometry': (type(None), dict)
        }
        zip_path = data_file_path('catalog.zip')
        out = read_stac(zip_path)

        # check for correctness of format
        self.assertIsInstance(out, list)
        for o in out:
            self.assertIsInstance(o, dict)
            for k, v in o.items():
                self.assertTrue(k in expected_keys)
                self.assertIsInstance(v, expected_keys[k])
            for uri in o['image_uris']:
                self.assertIsInstance(uri, str)

        # check for correctness of content (WRT the test catalog)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]['image_uris']), 1)


if __name__ == '__main__':
    unittest.main()
