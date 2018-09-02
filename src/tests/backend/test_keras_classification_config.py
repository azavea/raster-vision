import unittest
from tempfile import TemporaryDirectory
from google.protobuf import json_format
import json

import rastervision as rv
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg
from rastervision.utils.files import download_if_needed

from tests import data_file_path

CLASSES = ["car", "background"]

class TestKerasClassificationConfig(unittest.TestCase):
    template_uri = data_file_path("keras-classification/resnet50.json")


    def generate_task(self, classes=CLASSES, chip_size=300):
        return rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                            .with_classes(classes) \
                            .with_chip_size(chip_size) \
                            .build()

    def get_template_uri(self):
        return TestKerasClassificationConfig.template_uri

    def test_build_backend(self):
        b = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                            .with_task(self.generate_task()) \
                            .with_template(self.get_template_uri()) \
                            .with_batch_size(100) \
                            .build()

        self.assertEqual(b.kc_config["trainer"]["options"]["batchSize"], 100)
        self.assertListEqual(b.kc_config["trainer"]["options"]["classNames"],
                             CLASSES)

    def test_build_task_from_proto(self):
        config = {
            "backend_type": rv.KERAS_CLASSIFICATION,
            "keras_classification_config": {
                "kc_config": {
                    "model": {
                        "input_size": 300,
                        "type": "RESNET50",
                        "load_weights_by_name": False,
                        "model_path": ""
                    },
                    "trainer": {
                        "optimizer": {
                            "type": "ADAM",
                            "init_lr": 0.0001
                        },
                        "options": {
                            "training_data_dir": "",
                            "validation_data_dir": "",
                            "nb_epochs": 1,
                            "batch_size": 1,
                            "input_size": 300,
                            "output_dir": "",
                            "class_names": ["TEMPLATE"],
                            "short_epoch": True
                        }
                    }
                }
            }
        }

        msg = json_format.ParseDict(config, BackendConfigMsg())
        b = rv.BackendConfig.from_proto(msg)

        self.assertEqual(b.kc_config["model"]["type"], "RESNET50")

    def test_create_proto_from_backend(self):
        t = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                            .with_task(self.generate_task()) \
                            .with_template(self.get_template_uri()) \
                            .with_batch_size(100) \
                            .build()


        msg = t.to_proto()

        self.assertEqual(msg.backend_type, rv.KERAS_CLASSIFICATION)
        self.assertEqual(msg.keras_classification_config.kc_config["model"]["type"], "RESNET50")

    def test_requires_backend(self):
        with self.assertRaises(rv.ConfigError):
            b = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                                .with_task(self.generate_task()) \
                                .build()

    def test_copies_config_mods(self):
        bb1 = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                             .with_task(self.generate_task()) \
                             .with_template(self.get_template_uri()) \
                             .with_batch_size(100)

        bb2 = bb1.with_batch_size(200)

        b1 = bb1.build()
        b2 = bb2.build()

        self.assertEqual(b1.kc_config["trainer"]["options"]["batchSize"], 100)
        self.assertEqual(b2.kc_config["trainer"]["options"]["batchSize"], 200)

    def test_raise_error_on_no_backend_field(self):
        # Will raise since this backend template does not have num_steps
        with self.assertRaises(rv.ConfigError):
            b = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                                .with_task(self.generate_task()) \
                                .with_template(self.get_template_uri()) \
                                .with_batch_size(100) \
                                .with_num_epochs(100) \
                                .build()

    def test_with_config_fails_key_not_found(self):
        with self.assertRaises(rv.ConfigError):
            b = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                                .with_task(self.generate_task()) \
                                .with_template(self.get_template_uri()) \
                                .with_config({ "key_does_not_exist": 3 }) \
                                .build()

    def test_default_model_config(self):
        b = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                            .with_task(self.generate_task()) \
                            .with_model_defaults(rv.RESNET50_IMAGENET) \
                            .build()

        expected = ("https://github.com/fchollet/deep-learning-models/"
                    "releases/download/v0.2/"
                    "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
        self.assertEqual(b.pretrained_model_uri, expected)


if __name__ == "__main__":
    unittest.main()
