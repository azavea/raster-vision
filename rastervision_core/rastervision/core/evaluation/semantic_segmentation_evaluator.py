from typing import TYPE_CHECKING, Iterable, Iterator, Optional
import logging

from rastervision.core.data import ActivateMixin, RasterizedSource
from rastervision.core.data.vector_source import GeoJSONVectorSourceConfig
from rastervision.core.evaluation import (ClassificationEvaluator,
                                          SemanticSegmentationEvaluation)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rastervision.core.data import (
        Scene, ClassConfig, SemanticSegmentationLabels,
        SemanticSegmentationLabelSource, SemanticSegmentationLabelStore,
        VectorOutputConfig)


class SemanticSegmentationEvaluator(ClassificationEvaluator):
    """Evaluates predictions for a set of scenes."""

    def __init__(self,
                 class_config: 'ClassConfig',
                 output_uri: Optional[str] = None,
                 vector_output_uri: Optional[str] = None):
        super().__init__(class_config, output_uri)
        self.vector_output_uri = vector_output_uri

    def create_evaluation(self) -> SemanticSegmentationEvaluation:
        return SemanticSegmentationEvaluation(self.class_config)

    def process(self, scenes: Iterable['Scene'],
                tmp_dir: Optional[str] = None) -> None:
        """Override to also process vector predictions."""
        super().process(scenes)

        if self.vector_output_uri is not None:
            evaluation_global = self.create_evaluation()
            for scene in scenes:
                log.info(
                    f'Computing vector evaluation for scene {scene.id}...')
                evaluation = self.evaluate_scene_vector(scene)
                evaluation_global.merge(evaluation, scene_id=scene.id)
            evaluation_global.save(self.vector_output_uri)

    def evaluate_scene(self, scene: 'Scene') -> SemanticSegmentationEvaluation:
        """Override to pass null_class_id to filter_by_aoi()."""
        null_class_id = self.class_config.get_null_class_id()
        label_source: 'SemanticSegmentationLabelSource' = scene.label_source
        label_store: 'SemanticSegmentationLabelStore' = scene.label_store

        with ActivateMixin.compose(label_source, label_store):
            ground_truth = label_source.get_labels()
            predictions = label_store.get_labels()

        if scene.aoi_polygons:
            ground_truth = ground_truth.filter_by_aoi(scene.aoi_polygons,
                                                      null_class_id)
            predictions = predictions.filter_by_aoi(scene.aoi_polygons,
                                                    null_class_id)
        evaluation = self.evaluate_predictions(ground_truth, predictions)
        return evaluation

    def evaluate_scene_vector(
            self, scene: 'Scene') -> SemanticSegmentationEvaluation:
        label_source: 'SemanticSegmentationLabelSource' = scene.label_source
        label_store: 'SemanticSegmentationLabelStore' = scene.label_store

        has_vector_gt = isinstance(label_source.raster_source,
                                   RasterizedSource)
        has_vector_preds = label_store.vector_outputs is not None
        if not has_vector_gt:
            raise ValueError('Cannot evaluate vector predictions: '
                             'label source does not have vector ground truth.')
        if not has_vector_preds:
            raise ValueError('Cannot evaluate vector predictions: '
                             'label store does not have vector outputs.')

        with ActivateMixin.compose(label_source, label_store):
            gt_geojson = (
                label_source.raster_source.vector_source.get_geojson())
            if scene.aoi_polygons:
                gt_geojson = filter_geojson_by_aoi(gt_geojson,
                                                   scene.aoi_polygons)
            pred_geojsons = get_class_vector_preds(scene, label_store,
                                                   self.class_config)
            evaluation = self.evaluate_vector_predictions(
                gt_geojson, pred_geojsons, label_store.vector_outputs)
        return evaluation

    def evaluate_vector_predictions(
            self, gt_geojson: 'SemanticSegmentationLabels',
            pred_geojsons: Iterable[dict],
            vector_outputs: Iterable['VectorOutputConfig']
    ) -> SemanticSegmentationEvaluation:
        evaluation = self.create_evaluation()
        evaluation.compute_vector(gt_geojson, pred_geojsons, vector_outputs)
        return evaluation


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


def filter_geojson_by_aoi(geojson: dict, aoi_polygons: list) -> dict:
    from shapely.geometry import shape, mapping
    from shapely.strtree import STRtree

    # Note that this ignores class_id but that's ok because each prediction
    # GeoJSON file covers a single class_id. But, this may change in the
    # future.
    tree = STRtree([shape(f['geometry']) for f in geojson['features']])
    filtered_shapes = []
    for aoi_poly in aoi_polygons:
        shapes_in_aoi = tree.query(aoi_poly)
        for s in shapes_in_aoi:
            s_int = s.intersection(aoi_poly)
            filtered_shapes.append(s_int)

    features = [{
        'type': 'Feature',
        'geometry': mapping(s)
    } for s in filtered_shapes]

    return {'type': 'FeatureCollection', 'features': features}
