import unittest
import tempfile
import os
import json
from copy import deepcopy

from google.protobuf import json_format

from rastervision.utils.files import str_to_file, file_to_str
from rastervision.protos.predict_pb2 import PredictConfig
from rastervision.core.predict_package import (
    save_predict_package, load_predict_package, model_fn, stats_fn)


class TestPredictPackage(unittest.TestCase):
    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_dir_obj.name
        self.predict_package_uri = os.path.join(self.temp_dir, 'package.zip')

        # Save fake model and stats files.
        model_path = os.path.join(self.temp_dir, 'model')
        stats_path = os.path.join(self.temp_dir, 'stats.json')
        self.model_str = 'model123'
        self.stats_str = 'stats123'
        str_to_file(self.model_str, model_path)
        str_to_file(self.stats_str, stats_path)

        self.channel_order = [0, 1, 2]

        self.config_json = {
            "machine_learning": {
                "backend": "TF_OBJECT_DETECTION_API",
                "task": "OBJECT_DETECTION",
                "class_items": [{
                    "name": "car",
                    "id": 1
                }]
            },
            "options": {
                "model_uri": model_path,
                "object_detection_options": {
                    "score_thresh": 0.5,
                    "merge_thresh": 0.1
                },
                "chip_size": 300,
                "prediction_package_uri": self.predict_package_uri
            },
            "scenes": [{
                "prediction_label_store": {
                    "object_detection_geojson_file": {
                        "uri": "",
                    }
                },
                "raster_source": {
                    "geotiffFiles": {
                        "uris": []
                    },
                    "raster_transformer": {
                        "stats_uri": stats_path,
                        "channel_order": self.channel_order
                    }
                },
            }]
        }

        self.config = json_format.Parse(
            json.dumps(self.config_json), PredictConfig())

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test(self):
        # Use predict config linked to saved stats and model files to
        # save a package zip file. Then, load the package and test that
        # the resulting predict config is correct and the model and stats
        # files it references are in the right places.
        out_dir = os.path.join(self.temp_dir, 'output')
        labels_uri = 'labels_uri'
        image_uris = ['image_uri']

        save_predict_package(self.config)
        out_config = load_predict_package(
            self.predict_package_uri,
            out_dir,
            labels_uri,
            image_uris,
            channel_order=None)

        out_model_path = os.path.join(out_dir, 'package', model_fn)
        out_stats_path = os.path.join(out_dir, 'package', stats_fn)
        model_str = file_to_str(out_model_path)
        stats_str = file_to_str(out_stats_path)
        self.assertEqual(model_str, self.model_str)
        self.assertEqual(stats_str, self.stats_str)

        self.assertEqual(out_config.options.prediction_package_uri, '')
        self.assertEqual(out_config.options.model_uri, out_model_path)

        self.assertEqual(out_config.machine_learning,
                         self.config.machine_learning)
        self.assertEqual(out_config.options.object_detection_options,
                         self.config.options.object_detection_options)
        self.assertEqual(out_config.options.chip_size,
                         self.config.options.chip_size)

        self.assertEqual(len(out_config.scenes), 1)
        scene = out_config.scenes[0]
        self.assertEqual(
            scene.prediction_label_store.object_detection_geojson_file.uri,
            labels_uri)
        self.assertEqual(scene.raster_source.geotiff_files.uris, image_uris)
        self.assertEqual(scene.raster_source.raster_transformer.stats_uri,
                         out_stats_path)
        self.assertListEqual(
            list(scene.raster_source.raster_transformer.channel_order),
            self.channel_order)

    def test_channel_order(self):
        out_dir = os.path.join(self.temp_dir, 'output')
        labels_uri = 'labels_uri'
        image_uris = ['image_uri']

        save_predict_package(self.config)
        channel_order = [2, 1, 0]
        out_config = load_predict_package(
            self.predict_package_uri,
            out_dir,
            labels_uri,
            image_uris,
            channel_order=channel_order)

        scene = out_config.scenes[0]
        self.assertListEqual(
            list(scene.raster_source.raster_transformer.channel_order),
            channel_order)

    def test_ignores_no_ground_truth(self):
        out_dir = os.path.join(self.temp_dir, 'output')
        labels_uri = 'labels_uri'
        image_uris = ['image_uri']

        save_predict_package(self.config)
        out_config = load_predict_package(self.predict_package_uri, out_dir,
                                          labels_uri, image_uris)

        self.assertFalse(
            out_config.scenes[0].HasField('ground_truth_label_store'))

    def test_removes_ground_truth(self):
        print(os.path.curdir)
        out_dir = os.path.join(self.temp_dir, 'output')
        labels_uri = 'labels_uri'
        image_uris = ['image_uri']

        conf_json = deepcopy(self.config_json)
        conf_json['scenes'][0]['ground_truth_label_store'] = {}
        conf = json_format.Parse(json.dumps(conf_json), PredictConfig())

        save_predict_package(conf)
        out_config = load_predict_package(self.predict_package_uri, out_dir,
                                          labels_uri, image_uris)

        self.assertFalse(
            out_config.scenes[0].HasField('ground_truth_label_store'))


if __name__ == '__main__':
    unittest.main()
