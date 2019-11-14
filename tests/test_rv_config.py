import os
import unittest
import shutil
from unittest.mock import patch
import tempfile

from rastervision.rv_config import RVConfig


class TestRVConfig(unittest.TestCase):
    def setUp(self):
        self.prev_tmp = RVConfig.tmp_dir

    def tearDown(self):
        RVConfig.tmp_dir = self.prev_tmp

    def test_set_tmp_dir_explicit(self):
        RVConfig.tmp_dir = None
        tmp_dir = tempfile.TemporaryDirectory().name
        try:
            RVConfig.set_tmp_dir(tmp_dir)
            self.assertEqual(tmp_dir, RVConfig.get_tmp_dir_root())
            self.assertTrue(os.path.exists(tmp_dir))
            self.assertTrue(os.path.isdir(tmp_dir))
        finally:
            shutil.rmtree(tmp_dir)

    def test_set_tmp_dir_envvar(self):
        RVConfig.tmp_dir = None
        tmp_dir = tempfile.TemporaryDirectory().name
        with patch.dict(os.environ, {'TMPDIR': tmp_dir}, clear=True):
            try:
                RVConfig.set_tmp_dir()
                self.assertEqual(tmp_dir, RVConfig.get_tmp_dir_root())
                self.assertTrue(os.path.exists(tmp_dir))
                self.assertTrue(os.path.isdir(tmp_dir))
            finally:
                shutil.rmtree(tmp_dir)

    def test_set_tmp_dir_default(self):
        RVConfig.tmp_dir = None
        RVConfig.set_tmp_dir()
        self.assertEqual(RVConfig.DEFAULT_DIR, RVConfig.get_tmp_dir_root())
        self.assertTrue(os.path.exists(RVConfig.DEFAULT_DIR))
        self.assertTrue(os.path.isdir(RVConfig.DEFAULT_DIR))


if __name__ == '__main__':
    unittest.main()
