import unittest
from google.protobuf import json_format

import rastervision as rv
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg

from tests import data_file_path


class TestTFDeeplabConfig(unittest.TestCase):
    template_uri = data_file_path('tf_deeplab/mobilenet_v2.json')

    def generate_task(self, classes=['one', 'two'], chip_size=300):
        return rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_classes(classes) \
                            .with_chip_size(chip_size) \
                            .build()

    def get_template_uri(self):
        return TestTFDeeplabConfig.template_uri

    def test_build_backend(self):
        model_uri = 'pretrained_model'
        b = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                            .with_task(self.generate_task()) \
                            .with_template(self.get_template_uri()) \
                            .with_batch_size(100) \
                            .with_model_uri(model_uri) \
                            .with_fine_tune_checkpoint_name('foo') \
                            .build()

        self.assertEqual(b.tfdl_config['trainBatchSize'], 100)
        self.assertEqual(b.tfdl_config['modelVariant'], 'mobilenet_v2')
        self.assertEqual(b.model_uri, model_uri)
        self.assertEqual(b.fine_tune_checkpoint_name, 'foo')

    def test_build_backend_from_proto(self):
        config = {
            'backend_type': rv.TF_DEEPLAB,
            'tf_deeplab_config': {
                'tfdl_config': {
                    'decoderOutputStride': 2,
                    'outputStride': 17
                }
            }
        }

        msg = json_format.ParseDict(config, BackendConfigMsg())
        b = rv.BackendConfig.from_proto(msg)

        self.assertEqual(b.tfdl_config['decoderOutputStride'], 2)
        self.assertEqual(b.tfdl_config['outputStride'], 17)

    def test_create_proto_from_backend(self):
        t = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                            .with_task(self.generate_task()) \
                            .with_template(self.get_template_uri()) \
                            .with_batch_size(100) \
                            .with_fine_tune_checkpoint_name('foo') \
                            .build()

        msg = t.to_proto()

        self.assertEqual(msg.backend_type, rv.TF_DEEPLAB)
        self.assertEqual(msg.tf_deeplab_config.tfdl_config['trainBatchSize'],
                         100)
        self.assertEqual(msg.tf_deeplab_config.fine_tune_checkpoint_name,
                         'foo')

    def test_sets_fine_tune_checkpoint_to_experiment_name(self):
        task = self.generate_task()
        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
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
            rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                            .with_task(self.generate_task()) \
                            .build()

    def test_copies_config_mods(self):
        bb1 = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                              .with_task(self.generate_task()) \
                              .with_template(self.get_template_uri()) \
                              .with_batch_size(100)

        bb2 = bb1.with_batch_size(200)

        b1 = bb1.build()
        b2 = bb2.build()

        self.assertEqual(b1.tfdl_config['trainBatchSize'], 100)
        self.assertEqual(b2.tfdl_config['trainBatchSize'], 200)

    def test_with_config_fails_key_not_found(self):
        with self.assertRaises(rv.ConfigError):
            rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                            .with_task(self.generate_task()) \
                            .with_template(self.get_template_uri()) \
                            .with_config({'key_does_not_exist': 3}) \
                            .build()

    def test_config_missing_tfdl_config(self):
        with self.assertRaises(rv.ConfigError):
            rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                            .build()

    def test_default_model_config(self):
        b = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                            .with_task(self.generate_task()) \
                            .with_model_defaults(rv.MOBILENET_V2) \
                            .build()

        self.assertEqual(b.pretrained_model_uri,
                         ('http://download.tensorflow.org/models/'
                          'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'))


if __name__ == '__main__':
    unittest.main()
