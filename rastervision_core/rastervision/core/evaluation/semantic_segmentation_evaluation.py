from typing import TYPE_CHECKING, Any, Iterable, List, Union
import logging
import json

from sklearn.metrics import confusion_matrix
import numpy as np

from rastervision.core.evaluation import ClassEvaluationItem
from rastervision.core.evaluation import ClassificationEvaluation

if TYPE_CHECKING:
    from rastervision.core.data import (
        ClassConfig, SemanticSegmentationLabels, VectorOutputConfig)

log = logging.getLogger(__name__)


def is_geojson(data: Any) -> bool:
    if isinstance(data, dict):
        return True
    else:
        try:
            json.loads(data)
            retval = True
        except ValueError:
            retval = False
        return retval


class SemanticSegmentationEvaluation(ClassificationEvaluation):
    """Evaluation for semantic segmentation."""

    def __init__(self, class_config: 'ClassConfig'):
        super().__init__()
        self.class_config = class_config

    def compute(self, gt_labels: 'SemanticSegmentationLabels',
                pred_labels: 'SemanticSegmentationLabels') -> None:
        self.reset()

        # compute confusion matrix
        num_classes = len(self.class_config)
        labels = np.arange(num_classes)
        self.conf_mat = np.zeros((num_classes, num_classes))
        for window in pred_labels.get_windows():
            log.debug(f'Evaluating window: {window}')
            gt_arr = gt_labels.get_label_arr(window).ravel()
            pred_arr = pred_labels.get_label_arr(window).ravel()
            self.conf_mat += confusion_matrix(gt_arr, pred_arr, labels=labels)

        for class_id, class_name in enumerate(self.class_config.names):
            eval_item = ClassEvaluationItem.from_multiclass_conf_mat(
                conf_mat=self.conf_mat,
                class_id=class_id,
                class_name=class_name)
            self.class_to_eval_item[class_id] = eval_item

        self.compute_avg()

    def compute_vector(self, gt: Union[str, dict],
                       preds: Iterable[Union[str, dict]],
                       vector_outputs: Iterable['VectorOutputConfig']) -> None:
        """Compute evaluation over vector predictions.

        Args:
            gt (Union[str, dict]): Ground-truth GeoJSON. Either a string
                (containing unparsed GeoJSON or a file name), or a dictionary
                containing parsed GeoJSON.
            preds (Iterable[Union[str, dict]]): Prediction GeoJSONs. Either a
                string (containing unparsed GeoJSON or a file name), or a
                dictionary containing parsed GeoJSON.
            vector_outputs (Iterable[VectorOutputConfig]): VectorOutputConfig's
                corresponding to each prediction in preds.
        """
        import mask_to_polygons.vectorification as vectorification
        import mask_to_polygons.processing.score as score

        # Ground truth as list of geometries
        def get_geoms(x) -> List[dict]:
            if is_geojson(x):
                _x = x
                if 'features' in _x.keys():
                    _x = _x['features']
                geoms = []
                for feature in _x:
                    if 'geometry' in feature.keys():
                        geoms.append(feature['geometry'])
                    else:
                        geoms.append(feature)
            else:
                geoms = vectorification.geometries_from_geojson(x)

            return geoms

        gt = get_geoms(gt)
        if len(gt) == 0:
            return

        for pred, vo in zip(preds, vector_outputs):
            pred = get_geoms(pred)
            if len(pred) == 0:
                continue
            class_id = vo.class_id
            results = score.spacenet(pred, gt)
            eval_item = ClassEvaluationItem(
                class_id=class_id,
                class_name=self.class_config.names[class_id],
                tp=results['tp'],
                fp=results['fp'],
                fn=results['fn'],
                mode=vo.get_mode())
            self.class_to_eval_item[class_id] = eval_item

        self.compute_avg()
