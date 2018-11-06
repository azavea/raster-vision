import unittest

import rastervision as rv

from tests import data_file_path


class TestExperimentConfig(unittest.TestCase):
    @staticmethod
    def get_test_task():
        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({
                                'car': (1, 'blue'),
                                'building': (2, 'red')}) \
                            .with_chip_options(neg_ratio=0.0,
                                               ioa_thresh=1.0,
                                               window_method='sliding') \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

        return task

    def test_object_detection_exp(self):
        root_uri = '/some/dummy/root'
        img_path = '/dummy.tif'
        label_path = '/dummy.json'
        backend_conf_path = data_file_path(
            'tf_object_detection/'
            'embedded_ssd_mobilenet_v1_coco.config')

        pretrained_model = ('https://dummy.com/model.gz')

        task = self.get_test_task()

        backend = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                                  .with_task(task) \
                                  .with_template(backend_conf_path) \
                                  .with_pretrained_model(pretrained_model) \
                                  .with_train_options(sync_interval=None,
                                                      do_monitoring=False) \
                                  .build()

        raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                          .with_uri(img_path) \
                          .with_channel_order([0, 1, 2]) \
                          .with_stats_transformer() \
                          .build()

        scene = rv.SceneConfig.builder() \
                              .with_task(task) \
                              .with_id('od_test') \
                              .with_raster_source(raster_source) \
                              .with_label_source(label_path) \
                              .build()

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scene(scene) \
                                  .with_validation_scene(scene) \
                                  .build()

        analyzer = rv.analyzer.StatsAnalyzerConfig()

        e = rv.ExperimentConfig.builder() \
                               .with_id('object-detection-test') \
                               .with_root_uri(root_uri) \
                               .with_task(task) \
                               .with_backend(backend) \
                               .with_dataset(dataset) \
                               .with_analyzer(analyzer) \
                               .with_train_key('model_name') \
                               .build()

        msg = e.to_proto()
        e2 = rv.ExperimentConfig.from_proto(msg)

        self.assertEqual(e.train_uri, '/some/dummy/root/train/model_name')
        self.assertEqual(e.analyze_uri,
                         '/some/dummy/root/analyze/object-detection-test')

        self.assertEqual(e.analyze_uri, e2.analyze_uri)
        self.assertEqual(e.chip_uri, e2.chip_uri)
        self.assertEqual(e.train_uri, e2.train_uri)
        self.assertEqual(e.predict_uri, e2.predict_uri)
        self.assertEqual(e.eval_uri, e2.eval_uri)

        self.assertEqual(
            e2.dataset.train_scenes[0].label_source.vector_source.uri,
            '/dummy.json')
        self.assertEqual(
            e2.dataset.train_scenes[0].raster_source.channel_order, [0, 1, 2])

    def test_experiment_missing_configs_id(self):
        task = self.get_test_task()

        # missing ID
        with self.assertRaises(rv.ConfigError):
            rv.ExperimentConfig.builder()          \
                               .with_root_uri('')  \
                               .with_task(task)    \
                               .with_backend('')   \
                               .with_dataset('')   \
                               .with_analyzer('')  \
                               .with_train_uri('') \
                               .build()

    def test_experiment_missing_configs_backend(self):
        task = self.get_test_task()

        # missing backend
        with self.assertRaises(rv.ConfigError):
            rv.ExperimentConfig.builder()          \
                               .with_id('')        \
                               .with_root_uri('')  \
                               .with_task(task)    \
                               .with_dataset('')   \
                               .with_analyzer('')  \
                               .with_train_uri('') \
                               .build()

    def test_experiment_missing_train_key(self):
        task = self.get_test_task()
        # missing root_uri and other uris
        with self.assertRaises(rv.ConfigError):
            rv.ExperimentConfig.builder()          \
                               .with_id('')        \
                               .with_task(task)    \
                               .with_backend('')   \
                               .with_dataset('')   \
                               .build()

    def test_experiment_missing_multiple_configs(self):
        task = self.get_test_task()
        # missing root_uri and dataset and analyzer
        with self.assertRaises(rv.ConfigError):
            rv.ExperimentConfig.builder()           \
                               .with_id('')         \
                               .with_task(task)     \
                               .with_backend('')    \
                               .with_train_uri('')  \
                               .with_evaluators(['']) \
                               .build()

    def test_no_missing_config_max_with_root(self):
        task = self.get_test_task()
        # maximum args with root_uri
        try:
            rv.ExperimentConfig.builder()            \
                               .with_id('')          \
                               .with_root_uri('/dummy/root/uri')    \
                               .with_task(task)      \
                               .with_backend('')     \
                               .with_dataset('')     \
                               .with_evaluators([''])  \
                               .with_analyze_uri('') \
                               .with_chip_uri('')    \
                               .with_predict_uri('') \
                               .with_eval_uri('')    \
                               .with_bundle_uri('')  \
                               .build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')

    def test_no_missing_config_min_with_root(self):
        task = self.get_test_task()
        # minimum args with_root_uri
        try:
            rv.ExperimentConfig.builder()            \
                               .with_id('')          \
                               .with_evaluators([''])  \
                               .with_root_uri('/dummy/root/uri')    \
                               .with_task(task)      \
                               .with_backend('')     \
                               .with_dataset('')     \
                               .build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')


if __name__ == '__main__':
    unittest.main()
