import json

import numpy as np
import rasterio
import click

from object_detection.utils import object_detection_evaluation, label_map_util

from rv.commands.utils import (
    download_if_needed, make_temp_dir, get_local_path, upload_if_needed,
    get_boxes_from_geojson, download_and_build_vrt)
from rv.commands.settings import max_num_classes


def get_eval_result(ground_truth_path, predictions_path, image_dataset):
    gt_boxes, gt_box_to_class_id, _ = \
        get_boxes_from_geojson(ground_truth_path, image_dataset)
    gt_class_ids = np.array(
        [gt_box_to_class_id[box] for box in gt_boxes], dtype=int)
    # Subtract one because class id's start at 1, but evaluation api assumes
    # the start at 0. You might think we could just write the label_map.pbtxt
    # so the class ids start at 0, but that throws an exception.
    gt_class_ids -= 1
    gt_boxes = np.array(gt_boxes, dtype=float)

    pred_boxes, pred_box_to_class_id, pred_box_to_score = \
        get_boxes_from_geojson(predictions_path, image_dataset)
    pred_class_ids = np.array(
        [pred_box_to_class_id[box] for box in pred_boxes], dtype=int)
    pred_class_ids -= 1
    pred_scores = np.array(
        [pred_box_to_score[box] for box in pred_boxes], dtype=float)
    pred_boxes = np.array(pred_boxes, dtype=float)

    nb_gt_classes = len(set(gt_box_to_class_id.values()))
    od_eval = object_detection_evaluation.ObjectDetectionEvaluation(
        nb_gt_classes, matching_iou_threshold=0.1)
    image_key = 'image'
    od_eval.add_single_ground_truth_image_info(
        image_key, gt_boxes, gt_class_ids)
    od_eval.add_single_detected_image_info(
        image_key, pred_boxes, pred_scores, pred_class_ids)

    od_eval.evaluate()
    return od_eval.get_eval_result()


def write_results(output_path, label_map_path, eval_result):
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=max_num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    results = []
    for class_id in range(1, len(category_index) + 1):
        class_name = category_index[class_id]['name']
        # Get precision and recall assuming all boxes are used.
        # Subtract one to account for fact that class id's start at 1.
        precision = eval_result.precisions[class_id - 1][-1]
        recall = eval_result.recalls[class_id - 1][-1]
        class_results = {
            'name': class_name,
            'precision': precision,
            'recall': recall
        }
        results.append(class_results)

    with open(output_path, 'w') as output_file:
        output_file.write(json.dumps(results, indent=4))


@click.command()
@click.argument('image_uris', nargs=-1)
@click.argument('ground_truth_uri')
@click.argument('predictions_uri')
@click.argument('label_map_uri')
@click.argument('output_uri')
def eval_predictions(image_uris, ground_truth_uri, predictions_uri,
                     label_map_uri, output_uri):
    """Evaluate predictions against ground truth.

    Args:
        ground_truth_uri: GeoJSON file with ground truth bounding boxes
        predictions_uri: GeoJSON file with predicted bounding boxes
        output_uri: JSON file with metrics
    """
    temp_dir = '/opt/data/temp/'
    make_temp_dir(temp_dir)

    image_path = download_and_build_vrt(temp_dir, image_uris)
    image_dataset = rasterio.open(image_path)

    ground_truth_path = download_if_needed(temp_dir, ground_truth_uri)
    predictions_path = download_if_needed(temp_dir, predictions_uri)
    label_map_path = download_if_needed(temp_dir, label_map_uri)

    eval_result = get_eval_result(
        ground_truth_path, predictions_path, image_dataset)

    output_path = get_local_path(temp_dir, output_uri)
    write_results(output_path, label_map_path, eval_result)
    upload_if_needed(output_path, output_uri)


if __name__ == '__main__':
    eval_predictions()
