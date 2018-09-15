import unittest
from unittest import mock
import tempfile
from os.path import join

from rastervision.utils.rv_config import get_rv_config
from rastervision.utils.files import str_to_file


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_dir_obj.name
        self.env_config_path = join(self.temp_dir, 'env_config.ini')
        self.home_config_path = join(self.temp_dir, 'home_config.ini')

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_precedence(self):
        env_patches = {
            'RV_CONFIG_URI': self.env_config_path,
            'RV_GITHUB_REPO': 'c'
        }
        with mock.patch.dict('os.environ', env_patches):
            env_config_str = """
[default]
batch_job_def = a
"""
            str_to_file(env_config_str, self.env_config_path)

            home_config_str = """
[default]
batch_job_def = b
batch_job_queue = a
github_repo = a
"""
            str_to_file(home_config_str, self.home_config_path)

            config = get_rv_config(
                batch_job_queue='d',
                home_config_path=self.home_config_path)

            expected_config = {
                'batch_job_queue': 'd',
                'github_repo': 'c',
                'batch_job_def': 'b'
            }
            self.assertDictEqual(expected_config, config)
        # TODO set env vars back

    def test_no_files(self):
        env_patches = {
            'RV_GITHUB_REPO': 'a',
            'RV_BATCH_JOB_DEF': 'b'
        }
        with mock.patch.dict('os.environ', env_patches):
            config = get_rv_config(
                batch_job_queue='c',
                home_config_path=self.home_config_path)

            expected_config = {
                'batch_job_queue': 'c',
                'github_repo': 'a',
                'batch_job_def': 'b'
            }
            self.assertDictEqual(expected_config, config)

    def abstract_test_profile(self, profile):
        home_config_str = """
[a]
batch_job_def = a
batch_job_queue = a
github_repo = a
[b]
batch_job_def = b
batch_job_queue = b
github_repo = b
"""
        str_to_file(home_config_str, self.home_config_path)
        config = get_rv_config(
            home_config_path=self.home_config_path,
            profile=profile)

        expected_config = {
            'batch_job_queue': 'b',
            'github_repo': 'b',
            'batch_job_def': 'b'
        }
        self.assertDictEqual(expected_config, config)

    def test_profile(self):
        self.abstract_test_profile('b')

    def test_profile_in_env(self):
        env_patches = {
            'RV_PROFILE': 'b',
        }
        with mock.patch.dict('os.environ', env_patches):
            self.abstract_test_profile(None)


if __name__ == "__main__":
    unittest.main()
