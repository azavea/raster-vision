import unittest

from rastervision.pytorch_backend.examples.utils import read_stac

from tests import data_file_path


class TestUtils(unittest.TestCase):
    def test_read_stac(self):
        zip_path = data_file_path('catalog.zip')
        out = read_stac(zip_path)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], dict)
        self.assertEqual(len(out[0]['image_uris']), 1)


if __name__ == '__main__':
    unittest.main()
