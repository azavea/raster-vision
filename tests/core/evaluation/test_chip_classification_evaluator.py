import unittest
import os

from rastervision.pipeline import rv_config
from rastervision.pipeline.file_system import file_to_json
from rastervision.core.data import (
    ClassConfig, ChipClassificationLabelSourceConfig,
    GeoJSONVectorSourceConfig, ChipClassificationGeoJSONStoreConfig,
    RasterioSourceConfig, SceneConfig)
from rastervision.core.evaluation import (ChipClassificationEvaluatorConfig)

from tests import data_file_path


class TestChipClassificationEvaluator(unittest.TestCase):
    def test_accounts_for_aoi(self):
        class_config = ClassConfig(names=['car', 'building', 'background'])

        label_source_uri = data_file_path('evaluator/cc-label-filtered.json')
        label_source_cfg = ChipClassificationLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=label_source_uri, default_class_id=None))

        label_store_uri = data_file_path('evaluator/cc-label-full.json')
        label_store_cfg = ChipClassificationGeoJSONStoreConfig(
            uri=label_store_uri)

        raster_source_uri = data_file_path('evaluator/cc-label-img-blank.tif')
        raster_source_cfg = RasterioSourceConfig(uris=[raster_source_uri])

        aoi_uri = data_file_path('evaluator/cc-label-aoi.json')
        s = SceneConfig(
            id='test',
            raster_source=raster_source_cfg,
            label_source=label_source_cfg,
            label_store=label_store_cfg,
            aoi_uris=[aoi_uri])

        with rv_config.get_tmp_dir() as tmp_dir:
            scene = s.build(class_config, tmp_dir)
            output_uri = os.path.join(tmp_dir, 'eval.json')

            evaluator = ChipClassificationEvaluatorConfig(
                output_uri=output_uri).build(class_config)
            evaluator.process([scene], tmp_dir)

            overall = file_to_json(output_uri)['overall']
            for item in overall:
                self.assertEqual(item['f1'], 1.0)


if __name__ == '__main__':
    unittest.main()
