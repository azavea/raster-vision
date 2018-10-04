import os
import platform
import unittest
import shutil

from rastervision.rv_config import RVConfig


class TestRVConfig(unittest.TestCase):
    def setUp(self):
        self.prev_tmp = RVConfig.tmp_dir

    def tearDown(self):
        RVConfig.tmp_dir = self.prev_tmp

    def test_set_tmp_dir(self):
        if platform.system() == 'Linux':
            directory = '/tmp/xxx/'
            while os.path.exists(directory):
                directory = directory + 'xxx/'
            self.assertFalse(os.path.exists(directory))
            RVConfig.set_tmp_dir(directory)
            self.assertTrue(os.path.exists(directory))
            self.assertTrue(os.path.isdir(directory))
            shutil.rmtree(directory)
