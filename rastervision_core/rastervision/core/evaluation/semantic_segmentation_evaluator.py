from typing import TYPE_CHECKING, Iterable, Iterator
import logging

from shapely.geometry import shape, mapping
from shapely.strtree import STRtree

from rastervision.core.data import ActivateMixin, RasterizedSource
from rastervision.core.data.vector_source import GeoJSONVectorSourceConfig
from rastervision.core.evaluation import (ClassificationEvaluator,
                                          SemanticSegmentationEvaluation)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rastervision.core.data import (Scene, ClassConfig,
                                        SemanticSegmentationLabelSource,
                                        SemanticSegmentationLabelStore)


def filter_geojson_by_aoi(geojson: dict, aoi_polygons: list) -> dict:
    # Note that this ignores class_id but that's ok because each prediction GeoJSON file
    # covers a single class_id. But, this may change in the future.
    tree = STRtree([shape(f['geometry']) for f in geojson['features']])
    filtered_shapes = []
    for aoi_poly in aoi_polygons:
        shapes_in_aoi = tree.query(aoi_poly)
        for s in shapes_in_aoi:
            s_int = s.intersection(aoi_poly)
            filtered_shapes.append(s_int)

    features = [{
        'type': 'feature',
        'geometry': mapping(s)
    } for s in filtered_shapes]

    return {'type': 'FeatureCollection', 'features': features}


class SemanticSegmentationEvaluator(ClassificationEvaluator):
    """Evaluates predictions for a set of scenes.
    """

    def __init__(self, class_config: 'ClassConfig', output_uri: str,
                 vector_output_uri: str):
        super().__init__(class_config, output_uri)
        self.vector_output_uri = vector_output_uri

    def create_evaluation(self) -> SemanticSegmentationEvaluation:
        return SemanticSegmentationEvaluation(self.class_config)

    def process(self, scenes: Iterable['Scene'], tmp_dir: str) -> None:
        evaluation_global = self.create_evaluation()
        vect_evaluation_global = self.create_evaluation()
        null_class_id = self.class_config.get_null_class_id()

        for scene in scenes:
            log.info(f'Computing evaluation for scene {scene.id}...')
            label_source: 'SemanticSegmentationLabelSource' = (
                scene.ground_truth_label_source)
            label_store: 'SemanticSegmentationLabelStore' = (
                scene.prediction_label_store)
            with ActivateMixin.compose(label_source, label_store):
                # -----------
                # raster eval
                # -----------
                ground_truth = label_source.get_labels()
                predictions = label_store.get_labels()

                if scene.aoi_polygons:
                    # Filter labels based on AOI.
                    ground_truth = ground_truth.filter_by_aoi(
                        scene.aoi_polygons, null_class_id)
                    predictions = predictions.filter_by_aoi(
                        scene.aoi_polygons, null_class_id)
                evaluation_scene = self.create_evaluation()
                evaluation_scene.compute(ground_truth, predictions)
                evaluation_global.merge(evaluation_scene, scene_id=scene.id)

                # -----------
                # vector eval
                # -----------
                has_vector_gt = isinstance(label_source.raster_source,
                                           RasterizedSource)
                has_vector_preds = label_store.vector_outputs is not None
                if not (has_vector_gt and has_vector_preds):
                    continue

                gt_geojson = (
                    label_source.raster_source.vector_source.get_geojson())
                if scene.aoi_polygons:
                    gt_geojson = filter_geojson_by_aoi(gt_geojson,
                                                       scene.aoi_polygons)
                pred_geojsons = get_class_vector_preds(scene, label_store,
                                                       self.class_config)
                vect_evaluation_scene = self.create_evaluation()
                vect_evaluation_scene.compute_vector(
                    gt_geojson, pred_geojsons, label_store.vector_outputs)
                vect_evaluation_global.merge(
                    vect_evaluation_scene, scene_id=scene.id)

        if not evaluation_global.is_empty():
            evaluation_global.save(self.output_uri)
        if not vect_evaluation_global.is_empty():
            vect_evaluation_global.save(self.vector_output_uri)


def get_class_vector_preds(scene: 'Scene',
                           label_store: 'SemanticSegmentationLabelStore',
                           class_config: 'ClassConfig') -> Iterator[dict]:
    """Returns a generator that yields pred geojsons from
    label_store.vector_outputs."""
    class_ids = [vo.class_id for vo in label_store.vector_outputs]
    if len(set(class_ids)) < len(class_ids):
        raise ValueError('SemanticSegmentationEvaluator expects there to be '
                         'only one VectorOutputConfig per class.')
    for vo in label_store.vector_outputs:
        pred_geojson_uri = vo.uri
        class_id = vo.class_id
        pred_geojson_source_cfg = GeoJSONVectorSourceConfig(
            uri=pred_geojson_uri, default_class_id=class_id)
        pred_geojson_source = pred_geojson_source_cfg.build(
            class_config, scene.raster_source.get_crs_transformer())
        pred_geojson: dict = pred_geojson_source.get_geojson()

        if scene.aoi_polygons:
            pred_geojson = filter_geojson_by_aoi(pred_geojson,
                                                 scene.aoi_polygons)
        yield pred_geojson
