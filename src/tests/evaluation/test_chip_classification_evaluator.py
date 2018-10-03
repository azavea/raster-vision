import unittest
import os
import json

import rastervision as rv
from rastervision.rv_config import RVConfig

from tests import data_file_path


class TestChipClassificationEvaluator(unittest.TestCase):
    def test_accounts_for_aoi(self):
        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                            .with_classes(['car', 'building', 'background']) \
                            .build()

        label_source_uri = data_file_path('evaluator/cc-label-filtered.json')
        label_source = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                                           .with_uri(label_source_uri) \
                                           .build()

        label_store_uri = data_file_path('evaluator/cc-label-full.json')
        label_store = rv.LabelStoreConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                                         .with_uri(label_store_uri) \
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
                          .with_label_store(label_store) \
                          .with_aoi_uri(aoi_uri) \
                          .build()

        with RVConfig.get_tmp_dir() as tmp_dir:
            scene = s.create_scene(task, tmp_dir)

            output_uri = os.path.join(tmp_dir, 'eval.json')

            e = rv.EvaluatorConfig.builder(rv.CHIP_CLASSIFICATION_EVALUATOR) \
                                  .with_task(task) \
                                  .with_output_uri(output_uri) \
                                  .build()

            evaluator = e.create_evaluator()

            evaluator.process([scene], tmp_dir)

            results = None
            with open(output_uri) as f:
                results = json.loads(f.read())

            for result in results:
                self.assertEqual(result['f1'], 1.0)


if __name__ == '__main__':
    unittest.main()
