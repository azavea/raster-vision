import unittest

from rastervision.core.data import ClassConfig
from rastervision.core.data.vector_transformer import (
    ClassInferenceTransformer, ClassInferenceTransformerConfig)
from rastervision.core.data.utils import (geometries_to_geojson,
                                          geometry_to_feature)
from rastervision.core.data.vector_transformer.label_maker.filter import (
    create_filter)


def make_feature(**kwargs) -> dict:
    geometry = dict(type='Polygon', coordinates=[])
    return geometry_to_feature(geometry, properties=kwargs)


class TestClassInferenceTransformerConfig(unittest.TestCase):
    def test_build(self):
        cfg = ClassInferenceTransformerConfig()
        tf = cfg.build()
        self.assertIsInstance(tf, ClassInferenceTransformer)


class TestClassInferenceTransformer(unittest.TestCase):
    def test_init(self):
        args = dict(
            default_class_id=None,
            class_name_mapping=dict(old_name='new_name'))
        self.assertRaises(ValueError,
                          lambda: ClassInferenceTransformer(**args))

    def test_inference_with_class_id(self):
        feat = make_feature(class_id=1)
        class_id = ClassInferenceTransformer.infer_feature_class_id(
            feat, default_class_id=None)
        self.assertEqual(class_id, 1)

    def test_inference_with_class_name(self):
        feat = make_feature(class_name='bg')
        class_id = ClassInferenceTransformer.infer_feature_class_id(
            feat,
            default_class_id=None,
            class_config=ClassConfig(names=['bg', 'fg']))
        self.assertEqual(class_id, 0)

    def test_inference_with_label(self):
        feat = make_feature(label='bg')
        class_id = ClassInferenceTransformer.infer_feature_class_id(
            feat,
            default_class_id=None,
            class_config=ClassConfig(names=['bg', 'fg']))
        self.assertEqual(class_id, 0)

    def test_inference_with_class_name_mapping(self):
        feat = make_feature(class_name='old_name')
        args = dict(
            feature=feat,
            default_class_id=None,
            class_name_mapping=dict(old_name='new_name'))
        self.assertRaises(
            ValueError,
            lambda: ClassInferenceTransformer.infer_feature_class_id(**args))

        feat = make_feature(class_name='old_name')
        class_id = ClassInferenceTransformer.infer_feature_class_id(
            feat,
            default_class_id=None,
            class_config=ClassConfig(names=['bg', 'new_name']),
            class_name_mapping=dict(old_name='new_name'))
        self.assertEqual(class_id, 1)

        feat = make_feature(label='old_name')
        class_id = ClassInferenceTransformer.infer_feature_class_id(
            feat,
            default_class_id=None,
            class_config=ClassConfig(names=['bg', 'new_name']),
            class_name_mapping=dict(old_name='new_name'))
        self.assertEqual(class_id, 1)

    def test_inference_with_filter(self):
        class_id_to_filter = {
            2:
            create_filter(['==', 'type', 'building']),
            3:
            create_filter(
                ['any', ['==', 'type', 'car'], ['==', 'type', 'auto']])
        }

        feat = make_feature(type='building')
        class_id = ClassInferenceTransformer.infer_feature_class_id(
            feat, default_class_id=None, class_id_to_filter=class_id_to_filter)
        self.assertEqual(class_id, 2)

        feat = make_feature(type='car')
        class_id = ClassInferenceTransformer.infer_feature_class_id(
            feat, default_class_id=None, class_id_to_filter=class_id_to_filter)
        self.assertEqual(class_id, 3)

        feat = make_feature(type='auto')
        class_id = ClassInferenceTransformer.infer_feature_class_id(
            feat, default_class_id=None, class_id_to_filter=class_id_to_filter)
        self.assertEqual(class_id, 3)

    def test_inference_default(self):
        feat = make_feature()
        class_id = ClassInferenceTransformer.infer_feature_class_id(
            feat, default_class_id=None)
        self.assertIsNone(class_id)

        class_id = ClassInferenceTransformer.infer_feature_class_id(
            feat, default_class_id=0)
        self.assertEqual(class_id, 0)

    def test_transform(self):
        features_in = [
            make_feature(),
            make_feature(class_id=1),
            make_feature(class_name='bg'),
            make_feature(label='fg'),
            make_feature(type='building'),
            make_feature(type='car'),
            make_feature(type='auto'),
        ]
        geojson_in = geometries_to_geojson(features_in)

        tf = ClassInferenceTransformer(default_class_id=None)
        geojson_out = tf(geojson_in)
        features_out = geojson_out['features']
        self.assertEqual(len(features_out), 1)
        self.assertEqual(features_out[0]['properties']['class_id'], 1)

        tf = ClassInferenceTransformer(default_class_id=0)
        geojson_out = tf(geojson_in)
        features_out = geojson_out['features']
        class_ids_out = [f['properties']['class_id'] for f in features_out]
        self.assertEqual(len(features_out), len(features_in))
        self.assertListEqual(class_ids_out, [0, 1, 0, 0, 0, 0, 0])

        tf = ClassInferenceTransformer(
            default_class_id=0, class_config=ClassConfig(names=['bg', 'fg']))
        geojson_out = tf(geojson_in)
        features_out = geojson_out['features']
        class_ids_out = [f['properties']['class_id'] for f in features_out]
        self.assertEqual(len(features_out), len(features_in))
        self.assertListEqual(class_ids_out, [0, 1, 0, 1, 0, 0, 0])

        class_id_to_filter = {
            2: ['==', 'type', 'building'],
            3: ['any', ['==', 'type', 'car'], ['==', 'type', 'auto']]
        }
        tf = ClassInferenceTransformer(
            default_class_id=0,
            class_config=ClassConfig(names=['bg', 'fg']),
            class_id_to_filter=class_id_to_filter)
        geojson_out = tf(geojson_in)
        features_out = geojson_out['features']
        class_ids_out = [f['properties']['class_id'] for f in features_out]
        self.assertEqual(len(features_out), len(features_in))
        self.assertListEqual(class_ids_out, [0, 1, 0, 1, 2, 3, 3])


if __name__ == '__main__':
    unittest.main()
