import unittest

from rastervision.core.box import Box
from rastervision.core.data import ClassConfig, IdentityCRSTransformer
from rastervision.core.data.utils import geojson_to_geoms
from rastervision.core.data.label_store.utils import boxes_to_geojson


class TestUtils(unittest.TestCase):
    def test_boxes_to_geojson_without_scores(self):
        class_config = ClassConfig(names=['bg', 'fg'], null_class='bg')
        crs_transformer = IdentityCRSTransformer()
        boxes_in = [Box(0, 0, 100, 100), Box(0, 0, 200, 200)]
        class_ids_in = [0, 1]
        class_names_in = ['bg', 'fg']
        scores_in = None
        geojson = boxes_to_geojson(
            boxes=boxes_in,
            class_ids=class_ids_in,
            crs_transformer=crs_transformer,
            class_config=class_config,
            scores=scores_in)

        properties_out = [f['properties'] for f in geojson['features']]
        class_ids_out = [p.get('class_id') for p in properties_out]
        class_names_out = [p.get('class_name') for p in properties_out]
        scores_out_single = [p.get('score') for p in properties_out]
        scores_out_mult = [p.get('scores') for p in properties_out]
        self.assertListEqual(class_ids_out, class_ids_in)
        self.assertListEqual(class_names_out, class_names_in)
        self.assertListEqual(scores_out_single, [None, None])
        self.assertListEqual(scores_out_mult, [None, None])
        boxes_out = [Box.from_shapely(g) for g in geojson_to_geoms(geojson)]
        for box_in, box_out in zip(boxes_in, boxes_out):
            self.assertEqual(box_in, box_out)

    def test_boxes_to_geojson_with_scores(self):
        class_config = ClassConfig(names=['bg', 'fg'], null_class='bg')
        crs_transformer = IdentityCRSTransformer()
        boxes_in = [Box(0, 0, 100, 100), Box(0, 0, 200, 200)]
        class_ids_in = [0, 1]
        class_names_in = ['bg', 'fg']
        scores_in = [0.9, [0.2, 0.8]]
        geojson = boxes_to_geojson(
            boxes=boxes_in,
            class_ids=class_ids_in,
            crs_transformer=crs_transformer,
            class_config=class_config,
            scores=scores_in)

        properties_out = [f['properties'] for f in geojson['features']]
        class_ids_out = [p.get('class_id') for p in properties_out]
        class_names_out = [p.get('class_name') for p in properties_out]
        scores_out_single = [p.get('score') for p in properties_out]
        scores_out_mult = [p.get('scores') for p in properties_out]
        self.assertListEqual(class_ids_out, class_ids_in)
        self.assertListEqual(class_names_out, class_names_in)
        self.assertListEqual(scores_out_single, [0.9, None])
        self.assertListEqual(scores_out_mult, [None, [0.2, 0.8]])
        boxes_out = [Box.from_shapely(g) for g in geojson_to_geoms(geojson)]
        for box_in, box_out in zip(boxes_in, boxes_out):
            self.assertEqual(box_in, box_out)


if __name__ == '__main__':
    unittest.main()
