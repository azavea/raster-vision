import os
import unittest

import rastervision as rv
from rastervision.rv_config import RVConfig

from tests import data_file_path


class TestChipClassification(unittest.TestCase):
    def test_make_predict_windows_with_aoi(self):
        task_config = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                                   .with_chip_size(200) \
                                   .with_classes(['car', 'building', 'background']) \
                                   .build()

        backend_config = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                                         .with_task(task_config) \
                                         .with_model_defaults(rv.RESNET50_IMAGENET) \
                                         .with_pretrained_model(None) \
                                         .build()

        label_source_uri = data_file_path('evaluator/cc-label-full.json')
        label_source = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                                           .with_uri(label_source_uri) \
                                           .build()

        label_source_2_uri = data_file_path('evaluator/cc-label-filtered.json')
        label_source_2 = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                                           .with_uri(label_source_2_uri) \
                                           .build()

        source_uri = data_file_path('evaluator/cc-label-img-blank.tif')
        raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                             .with_uri(source_uri) \
                                             .build()

        aoi_uri = data_file_path('evaluator/cc-label-aoi.json')
        s = rv.SceneConfig.builder() \
                          .with_id('test') \
                          .with_raster_source(raster_source) \
                          .with_label_source(label_source) \
                          .with_aoi_uri(aoi_uri) \
                          .build()

        with RVConfig.get_tmp_dir() as tmp_dir:
            scene = s.create_scene(task_config, tmp_dir)
            backend = backend_config.create_backend(task_config)
            task = task_config.create_task(backend)

            with scene.activate():
                windows = task.get_train_windows(scene)

            from rastervision.data import (ChipClassificationLabels,
                                           ChipClassificationGeoJSONStore)
            labels = ChipClassificationLabels()
            for w in windows:
                labels.set_cell(w, 1)
            store = ChipClassificationGeoJSONStore(
                os.path.join(tmp_dir, 'test.json'),
                scene.raster_source.get_crs_transformer(),
                task_config.class_map)
            store.save(labels)

            ls = label_source_2.create_source(
                task_config, scene.raster_source.get_extent(),
                scene.raster_source.get_crs_transformer(), tmp_dir)
            actual = ls.get_labels().get_cells()

            self.assertEqual(len(windows), len(actual))


if __name__ == '__main__':
    unittest.main()
