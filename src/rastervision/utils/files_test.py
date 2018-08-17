import tempfile
import os
import unittest
import json

import boto3
from moto import mock_s3

from rastervision.utils.files import (
    file_to_str, str_to_file, download_if_needed, upload_if_needed,
    NotReadableError, NotWritableError, load_json_config,
    ProtobufParseException, make_dir, get_local_path)
from rastervision.protos.machine_learning_pb2 import MachineLearning


class TestMakeDir(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_args(self):
        dir = os.path.join(self.temp_dir.name, 'hello')
        make_dir(dir)
        self.assertTrue(os.path.isdir(dir))

    def test_check_empty(self):
        path = os.path.join(self.temp_dir.name, 'hello', 'hello.txt')
        dir = os.path.dirname(path)
        str_to_file('hello', path)

        make_dir(dir, check_empty=False)
        with self.assertRaises(Exception):
            make_dir(dir, check_empty=True)

    def test_force_empty(self):
        path = os.path.join(self.temp_dir.name, 'hello', 'hello.txt')
        dir = os.path.dirname(path)
        str_to_file('hello', path)

        make_dir(dir, force_empty=False)
        self.assertTrue(os.path.isfile(path))
        make_dir(dir, force_empty=True)
        is_empty = len(os.listdir(dir)) == 0
        self.assertTrue(is_empty)

    def test_use_dirname(self):
        path = os.path.join(self.temp_dir.name, 'hello', 'hello.txt')
        dir = os.path.dirname(path)
        make_dir(path, use_dirname=True)
        self.assertTrue(os.path.isdir(dir))


class TestGetLocalPath(unittest.TestCase):
    def test_local(self):
        download_dir = '/download_dir'
        uri = '/my/file.txt'
        path = get_local_path(uri, download_dir)
        self.assertEqual(path, uri)

    def test_s3(self):
        download_dir = '/download_dir'
        uri = 's3://bucket/my/file.txt'
        path = get_local_path(uri, download_dir)
        self.assertEqual(path, '/download_dir/s3/bucket/my/file.txt')

    def test_http(self):
        download_dir = '/download_dir'
        uri = 'http://bucket/my/file.txt'
        path = get_local_path(uri, download_dir)
        self.assertEqual(path, '/download_dir/http/bucket/my/file.txt')


class TestFileToStr(unittest.TestCase):
    """Test file_to_str and str_to_file."""

    def setUp(self):
        # Setup mock S3 bucket.
        self.mock_s3 = mock_s3()
        self.mock_s3.start()
        self.s3 = boto3.client('s3')
        self.bucket_name = 'mock_bucket'
        self.s3.create_bucket(Bucket=self.bucket_name)

        self.content_str = 'hello'
        self.file_name = 'hello.txt'
        self.s3_path = 's3://{}/{}'.format(self.bucket_name, self.file_name)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.local_path = os.path.join(self.temp_dir.name, self.file_name)

    def tearDown(self):
        self.temp_dir.cleanup()
        self.mock_s3.stop()

    def test_file_to_str_local(self):
        str_to_file(self.content_str, self.local_path)
        content_str = file_to_str(self.local_path)
        self.assertEqual(self.content_str, content_str)

        wrong_path = '/wrongpath/x.txt'
        with self.assertRaises(NotReadableError):
            file_to_str(wrong_path)

    def test_file_to_str_s3(self):
        wrong_path = 's3://wrongpath/x.txt'

        with self.assertRaises(NotWritableError):
            str_to_file(self.content_str, wrong_path)

        str_to_file(self.content_str, self.s3_path)
        content_str = file_to_str(self.s3_path)
        self.assertEqual(self.content_str, content_str)

        with self.assertRaises(NotReadableError):
            file_to_str(wrong_path)


class TestDownloadIfNeeded(unittest.TestCase):
    """Test download_if_needed and upload_if_needed and str_to_file."""

    def setUp(self):
        # Setup mock S3 bucket.
        self.mock_s3 = mock_s3()
        self.mock_s3.start()
        self.s3 = boto3.client('s3')
        self.bucket_name = 'mock_bucket'
        self.s3.create_bucket(Bucket=self.bucket_name)

        self.content_str = 'hello'
        self.file_name = 'hello.txt'
        self.s3_path = 's3://{}/{}'.format(self.bucket_name, self.file_name)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.local_path = os.path.join(self.temp_dir.name, self.file_name)

    def tearDown(self):
        self.temp_dir.cleanup()
        self.mock_s3.stop()

    def test_download_if_needed_local(self):
        with self.assertRaises(NotReadableError):
            download_if_needed(self.local_path, self.temp_dir.name)

        str_to_file(self.content_str, self.local_path)
        upload_if_needed(self.local_path, self.local_path)
        local_path = download_if_needed(self.local_path, self.temp_dir.name)
        self.assertEqual(local_path, self.local_path)

    def test_download_if_needed_s3(self):
        with self.assertRaises(NotReadableError):
            download_if_needed(self.s3_path, self.temp_dir.name)

        str_to_file(self.content_str, self.local_path)
        upload_if_needed(self.local_path, self.s3_path)
        local_path = download_if_needed(self.s3_path, self.temp_dir.name)
        content_str = file_to_str(local_path)
        self.assertEqual(self.content_str, content_str)

        wrong_path = 's3://wrongpath/x.txt'
        with self.assertRaises(NotWritableError):
            upload_if_needed(local_path, wrong_path)


class TestLoadJsonConfig(unittest.TestCase):
    def setUp(self):
        self.file_name = 'config.json'
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, self.file_name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_config_file(self, config):
        file_contents = json.dumps(config)
        with open(self.file_path, 'w') as myfile:
            myfile.write(file_contents)

    def test_valid(self):
        config = {
            'task': 'CLASSIFICATION',
            'backend': 'KERAS_CLASSIFICATION',
            'class_items': [{
                'id': 1,
                'name': 'car'
            }]
        }
        self.write_config_file(config)
        ml_config = load_json_config(self.file_path, MachineLearning())
        self.assertEqual(ml_config.task,
                         MachineLearning.Backend.Value('KERAS_CLASSIFICATION'))
        self.assertEqual(ml_config.backend,
                         MachineLearning.Task.Value('CLASSIFICATION'))
        self.assertEqual(ml_config.class_items[0].id, 1)
        self.assertEqual(ml_config.class_items[0].name, 'car')
        self.assertEqual(len(ml_config.class_items), 1)

    def test_bogus_field(self):
        config = {
            'task': 'CLASSIFICATION',
            'backend': 'KERAS_CLASSIFICATION',
            'class_items': [{
                'id': 1,
                'name': 'car'
            }],
            'bogus_field': 0
        }

        self.write_config_file(config)
        with self.assertRaises(ProtobufParseException):
            load_json_config(self.file_path, MachineLearning())

    def test_bogus_value(self):
        config = {'task': 'bogus_value'}
        self.write_config_file(config)
        with self.assertRaises(ProtobufParseException):
            load_json_config(self.file_path, MachineLearning())

    def test_invalid_json(self):
        invalid_json_str = '''
            {
                "task": "CLASSIFICATION
            }
        '''
        with open(self.file_path, 'w') as myfile:
            myfile.write(invalid_json_str)

        with self.assertRaises(ProtobufParseException):
            load_json_config(self.file_path, MachineLearning())


if __name__ == '__main__':
    unittest.main()
