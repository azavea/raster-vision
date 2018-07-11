import tempfile
import os
import unittest
import json

import boto3
from moto import mock_s3

from rastervision.utils.files import (
    file_to_str, NotFoundException, load_json_config, ProtobufParseException)
from rastervision.protos.machine_learning_pb2 import MachineLearning


class TestFileToStr(unittest.TestCase):
    def setUp(self):
        self.mock_s3 = mock_s3()
        self.mock_s3.start()

        # Save temp file.
        self.file_name = 'hello.txt'
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, self.file_name)
        self.file_contents = 'hello'
        with open(self.file_path, 'w') as myfile:
            myfile.write(self.file_contents)

        # Upload file to mock S3 bucket.
        self.s3 = boto3.client('s3')
        self.bucket_name = 'mock_bucket'
        self.s3.create_bucket(Bucket=self.bucket_name)
        self.s3_path = 's3://{}/{}'.format(self.bucket_name, self.file_name)
        self.s3.upload_file(
            self.file_path, self.bucket_name, self.file_name)

    def tearDown(self):
        self.temp_dir.cleanup()
        self.mock_s3.stop()

    def test_s3(self):
        str = file_to_str(self.s3_path)
        self.assertEqual(str, self.file_contents)

    def test_local(self):
        str = file_to_str(self.file_path)
        self.assertEqual(str, self.file_contents)

    def test_wrong_s3(self):
        wrong_path = 's3://{}/{}'.format(self.bucket_name, 'x.txt')
        with self.assertRaises(NotFoundException):
            file_to_str(wrong_path)

    def test_wrong_local(self):
        wrong_path = '/wrongpath/x.txt'
        with self.assertRaises(NotFoundException):
            file_to_str(wrong_path)


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
            'class_items': [
                {
                    'id': 1,
                    'name': 'car'
                }
            ]
        }
        self.write_config_file(config)
        ml_config = load_json_config(self.file_path, MachineLearning())
        self.assertEqual(
            ml_config.task,
            MachineLearning.Backend.Value('KERAS_CLASSIFICATION'))
        self.assertEqual(
            ml_config.backend,
            MachineLearning.Task.Value('CLASSIFICATION'))
        self.assertEqual(ml_config.class_items[0].id, 1)
        self.assertEqual(ml_config.class_items[0].name, 'car')
        self.assertEqual(len(ml_config.class_items), 1)

    def test_bogus_field(self):
        config = {
            'task': 'CLASSIFICATION',
            'backend': 'KERAS_CLASSIFICATION',
            'class_items': [
                {
                    'id': 1,
                    'name': 'car'
                }
            ],
            'bogus_field': 0
        }

        self.write_config_file(config)
        with self.assertRaises(ProtobufParseException):
            load_json_config(self.file_path, MachineLearning())

    def test_bogus_value(self):
        config = {
            'task': 'bogus_value'
        }
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
