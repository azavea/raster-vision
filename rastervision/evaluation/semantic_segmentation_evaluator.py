import logging
import json

from shapely.geometry import mapping
import shapely

from rastervision.data import ActivateMixin, geojson_to_shapes
from rastervision.utils.files import (file_to_str)
from rastervision.evaluation import (ClassificationEvaluator,
                                     SemanticSegmentationEvaluation)

log = logging.getLogger(__name__)


def filter_geojson_by_aoi(geojson, crs_transformer, aoi_polygons):
    shapes = [s for s, c in geojson_to_shapes(geojson, crs_transformer)]
    tree = shapely.strtree.STRtree(shapes)
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

    def __init__(self, class_map, output_uri, vector_output_uri):
        super().__init__(class_map, output_uri)
        self.vector_output_uri = vector_output_uri

    def create_evaluation(self):
        return SemanticSegmentationEvaluation(self.class_map)

    def process(self, scenes, tmp_dir):
        evaluation = self.create_evaluation()
        vect_evaluation = self.create_evaluation()

        for scene in scenes:
            log.info('Computing evaluation for scene {}...'.format(scene.id))
            label_source = scene.ground_truth_label_source
            label_store = scene.prediction_label_store
            with ActivateMixin.compose(label_source, label_store):
                ground_truth = label_source.get_labels()
                predictions = label_store.get_labels()

                if scene.aoi_polygons:
                    # Filter labels based on AOI.
                    ground_truth = ground_truth.filter_by_aoi(
                        scene.aoi_polygons)
                    predictions = predictions.filter_by_aoi(scene.aoi_polygons)
                scene_evaluation = self.create_evaluation()
                scene_evaluation.compute(ground_truth, predictions)
                evaluation.merge(scene_evaluation, scene_id=scene.id)

            if hasattr(label_source, 'source') and hasattr(
                    label_source.source, 'vector_source') and hasattr(
                        label_store, 'vector_output'):
                gt_geojson = label_source.source.vector_source.get_geojson()
                for vo in label_store.vector_output:
                    pred_geojson_uri = vo['uri']
                    mode = vo['mode']
                    class_id = vo['class_id']
                    pred_geojson = json.loads(file_to_str(pred_geojson_uri))

                    if scene.aoi_polygons:
                        gt_geojson = filter_geojson_by_aoi(
                            gt_geojson,
                            scene.raster_source.get_crs_transformer(),
                            scene.aoi_polygons)
                        pred_geojson = filter_geojson_by_aoi(
                            pred_geojson,
                            scene.raster_source.get_crs_transformer(),
                            scene.aoi_polygons)

                    vect_scene_evaluation = self.create_evaluation()
                    vect_scene_evaluation.compute_vector(
                        gt_geojson, pred_geojson, mode, class_id)
                    vect_evaluation.merge(
                        vect_scene_evaluation, scene_id=scene.id)

        if not evaluation.is_empty():
            evaluation.save(self.output_uri)
        if not vect_evaluation.is_empty():
            vect_evaluation.save(self.vector_output_uri)
