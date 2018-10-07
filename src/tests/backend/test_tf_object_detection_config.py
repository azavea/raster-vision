import unittest
from google.protobuf import json_format

import rastervision as rv
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg

from tests import data_file_path


class TestTFObjectDetectionConfig(unittest.TestCase):
    template_uri = data_file_path(
        'tf_object_detection/embedded_ssd_mobilenet_v1_coco.config')

    def generate_task(self, classes=['one', 'two'], chip_size=300):
        return rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_classes(classes) \
                            .with_chip_size(chip_size) \
                            .build()

    def get_template_uri(self):
        return TestTFObjectDetectionConfig.template_uri

    def test_build_backend(self):
        b = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                            .with_task(self.generate_task()) \
                            .with_template(self.get_template_uri()) \
                            .with_batch_size(100) \
                            .with_fine_tune_checkpoint_name('foo') \
                            .build()

        self.assertEqual(b.tfod_config['trainConfig']['batchSize'], 100)
        self.assertEqual(b.tfod_config['model']['ssd']['numClasses'], 2)
        self.assertEqual(b.fine_tune_checkpoint_name, 'foo')

    def test_build_task_from_proto(self):
        config = {
            'backend_type': rv.TF_OBJECT_DETECTION,
            'tf_object_detection_config': {
                'tfod_config': {
                    'model': {
                        'dummy': {
                            'numClasses': 5,
                            'imageResizer': {
                                'keepAspectRatioResizer': {
                                    'minDimension': 300,
                                    'maxDimension': 300
                                }
                            }
                        }
                    },
                    'trainConfig': {
                        'batchSize': 1,
                        'numSteps': 200
                    }
                }
            }
        }

        msg = json_format.ParseDict(config, BackendConfigMsg())
        b = rv.BackendConfig.from_proto(msg)

        self.assertEqual(b.tfod_config['trainConfig']['batchSize'], 1)
        self.assertEqual(b.tfod_config['trainConfig']['numSteps'], 200)

    def test_create_proto_from_backend(self):
        t = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                            .with_task(self.generate_task()) \
                            .with_template(self.get_template_uri()) \
                            .with_batch_size(100) \
                            .with_fine_tune_checkpoint_name('foo') \
                            .build()

        msg = t.to_proto()

        self.assertEqual(msg.backend_type, rv.TF_OBJECT_DETECTION)
        self.assertEqual(
            msg.tf_object_detection_config.tfod_config['trainConfig'][
                'batchSize'], 100)
        self.assertEqual(
            msg.tf_object_detection_config.fine_tune_checkpoint_name, 'foo')

    def test_sets_fine_tune_checkpoint_to_experiment_name(self):
        task = self.generate_task()
        backend = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                                  .with_task(task) \
                                  .with_template(self.get_template_uri()) \
                                  .with_batch_size(100) \
                                  .build()
        dataset = rv.DatasetConfig.builder().build()

        e = rv.ExperimentConfig.builder() \
                               .with_task(task) \
                               .with_backend(backend) \
                               .with_dataset(dataset) \
                               .with_id('foo-exp') \
                               .with_root_uri('.') \
                               .build()

        resolved_e = e.fully_resolve()

        self.assertEqual(resolved_e.backend.fine_tune_checkpoint_name,
                         'foo-exp')

    def test_requires_backend(self):
        with self.assertRaises(rv.ConfigError):
            rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                            .with_task(self.generate_task()) \
                            .build()

    def test_copies_config_mods(self):
        bb1 = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                              .with_task(self.generate_task()) \
                              .with_template(self.get_template_uri()) \
                              .with_batch_size(100)

        bb2 = bb1.with_batch_size(200)

        b1 = bb1.build()
        b2 = bb2.build()

        self.assertEqual(b1.tfod_config['trainConfig']['batchSize'], 100)
        self.assertEqual(b2.tfod_config['trainConfig']['batchSize'], 200)

    def test_raise_error_on_no_backend_field(self):
        with self.assertRaises(rv.ConfigError):
            rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                            .with_task(self.generate_task()) \
                            .with_template(self.get_template_uri()) \
                            .with_batch_size(100) \
                            .with_config({'bogusField': 1}) \
                            .build()

    def test_with_config_fails_key_not_found(self):
        with self.assertRaises(rv.ConfigError):
            rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                            .with_task(self.generate_task()) \
                            .with_template(self.get_template_uri()) \
                            .with_config({'key_does_not_exist': 3}) \
                            .build()

    def test_config_missing_tfod_config(self):
        with self.assertRaises(rv.ConfigError):
            rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                            .build()

    def test_sets_pretrained_model(self):
        pretrained_model_uri = ('http://download.tensorflow.org/'
                                'models/object_detection/'
                                'ssd_mobilenet_v1_coco_2017_11_17.tar.gz')
        b = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                            .with_task(self.generate_task()) \
                            .with_template(self.get_template_uri()) \
                            .with_batch_size(100) \
                            .with_pretrained_model(pretrained_model_uri) \
                            .build()

        self.assertEqual(b.tfod_config['trainConfig']['fineTuneCheckpoint'],
                         pretrained_model_uri)

    def test_default_model_config(self):
        b = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                            .with_task(self.generate_task()) \
                            .with_model_defaults(rv.SSD_MOBILENET_V1_COCO) \
                            .build()

        self.assertEqual(b.pretrained_model_uri,
                         ('http://download.tensorflow.org/'
                          'models/object_detection/'
                          'ssd_mobilenet_v1_coco_2017_11_17.tar.gz'))


if __name__ == '__main__':
    unittest.main()
