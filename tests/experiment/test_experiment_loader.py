import unittest
import os

import rastervision as rv
from rastervision.experiment import ExperimentLoader

from tests import data_file_path


class DummyExperimentSet(rv.ExperimentSet):
    def get_base(self):
        root_uri = '/some/dummy/root'
        img_path = '/dummy.tif'
        label_path = '/dummy.json'
        backend_conf_path = data_file_path(
            'tf_object_detection/'
            'embedded_ssd_mobilenet_v1_coco.config')

        pretrained_model = ('https://dummy.com/model.gz')

        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({
                                'car': (1, 'blue'),
                                'building': (2, 'red')
                            }) \
                            .with_chip_options(neg_ratio=0.0,
                                               ioa_thresh=1.0,
                                               window_method='sliding') \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

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

        return rv.ExperimentConfig.builder() \
                                  .with_root_uri(root_uri) \
                                  .with_task(task) \
                                  .with_backend(backend) \
                                  .with_dataset(dataset) \
                                  .with_analyzer(analyzer) \
                                  .with_train_key('model_name')

    def exp_experiment_1(self):
        return self.get_base() \
                   .with_id('experiment_1') \
                   .build()

    def exp_experiment_2(self, required_param):
        es = []
        for i in range(0, 2):
            es.append(self.get_base().with_id('experiment_{}_{}'.format(
                i + 1, required_param)).build())
        return es


class TestExperimentConfig(unittest.TestCase):
    def test_load_module(self):
        args = {'required_param': 'yes', 'dummy': 1}
        loader = ExperimentLoader(experiment_args=args)
        experiments = loader.load_from_module(__name__)
        self.assertEqual(len(experiments), 3)
        e_names = set(map(lambda e: e.id, experiments))
        self.assertEqual(
            e_names,
            set(['experiment_1', 'experiment_1_yes', 'experiment_2_yes']))

    def test_load_file(self):
        path = os.path.abspath(__file__)
        args = {'required_param': 'yes', 'dummy': 1}
        loader = ExperimentLoader(experiment_args=args)
        experiments = loader.load_from_file(path)
        self.assertEqual(len(experiments), 3)
        e_names = set(map(lambda e: e.id, experiments))
        self.assertEqual(
            e_names,
            set(['experiment_1', 'experiment_1_yes', 'experiment_2_yes']))

    def test_filter_module_by_method(self):
        name = '*2'
        args = {'required_param': 'x'}
        loader = ExperimentLoader(
            experiment_args=args, experiment_method_patterns=[name])
        experiments = loader.load_from_module(__name__)
        e_names = set(map(lambda e: e.id, experiments))
        self.assertEqual(e_names, set(['experiment_1_x', 'experiment_2_x']))

    def test_filter_module_by_name(self):
        name = '*2*y*'
        args = {'required_param': 'yes'}
        loader = ExperimentLoader(
            experiment_args=args, experiment_name_patterns=[name])
        experiments = loader.load_from_module(__name__)
        e_names = set(map(lambda e: e.id, experiments))
        self.assertEqual(e_names, set(['experiment_2_yes']))


if __name__ == '__main__':
    unittest.main()
