import json
from os.path import join

import rasterio
import click

from object_detection.utils import object_detection_evaluation, label_map_util

from rv.utils.files import (
    download_if_needed, make_dir, get_local_path, upload_if_needed,
    MyTemporaryDirectory)
from rv.utils.geo import get_boxes_from_geojson, download_and_build_vrt
from rv.detection.commands.settings import max_num_classes, temp_root_dir


def get_od_eval(ground_truth_path, predictions_path, image_dataset):
    gt_boxes, gt_classes, _ = \
        get_boxes_from_geojson(ground_truth_path, image_dataset)
    # Subtract one because class id's start at 1, but evaluation api assumes
    # the start at 0. You might think we could just write the label_map.pbtxt
    # so the class ids start at 0, but that throws an exception.
    gt_classes -= 1

    pred_boxes, pred_classes, pred_scores = \
        get_boxes_from_geojson(predictions_path, image_dataset)
    pred_classes -= 1

    nb_gt_classes = len(set(gt_classes))
    matching_iou_threshold = 0.5
    od_eval = object_detection_evaluation.ObjectDetectionEvaluation(
        nb_gt_classes, matching_iou_threshold=matching_iou_threshold)
    image_key = 'image'
    od_eval.add_single_ground_truth_image_info(
        image_key, gt_boxes, gt_classes)
    od_eval.add_single_detected_image_info(
        image_key, pred_boxes, pred_scores, pred_classes)

    od_eval.evaluate()
    return od_eval


def write_results(output_path, label_map_path, od_eval):
    make_dir(output_path, use_dirname=True)

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=max_num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    results = []
    for class_id in range(1, len(category_index) + 1):
        class_name = category_index[class_id]['name']
        # Subtract one to account for fact that class id's start at 1.
        # precisions and recalls are lists with one element for each
        # predicted box, assuming they are sorted by score. Each element is
        # the precision or recall assuming that all predicted boxes with that
        # score or above are used. So, the last element is the value assuming
        # that all predictions are used.
        eval_result = od_eval.get_eval_result()
        precisions = eval_result.precisions[class_id - 1]
        recalls = eval_result.recalls[class_id - 1]
        # Get precision and recall assuming all predicted boxes are used.
        precision = precisions[-1]
        recall = recalls[-1]
        f1 = (2 * precision * recall) / (precision + recall)

        gt_count = od_eval.num_gt_instances_per_class[class_id -1]
        pred_count = len(recalls)
        count_error = pred_count - gt_count
        norm_count_error = count_error / gt_count

        class_results = {
            'name': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'norm_count_error': norm_count_error
        }
        results.append(class_results)

    with open(output_path, 'w') as output_file:
        output_file.write(json.dumps(results, indent=4, sort_keys=True))


def _eval_predictions(image_uris, label_map_uri, ground_truth_uri,
                      predictions_uri, output_uri, save_temp):
    prefix = temp_root_dir
    temp_dir = join(prefix, 'eval-predictions') if save_temp else None
    with MyTemporaryDirectory(temp_dir, prefix) as temp_dir:
        image_path = download_and_build_vrt(image_uris, temp_dir)
        image_dataset = rasterio.open(image_path)

        ground_truth_path = download_if_needed(ground_truth_uri, temp_dir)
        predictions_path = download_if_needed(predictions_uri, temp_dir)
        label_map_path = download_if_needed(label_map_uri, temp_dir)

        od_eval = get_od_eval(
            ground_truth_path, predictions_path, image_dataset)

        output_path = get_local_path(output_uri, temp_dir)
        write_results(output_path, label_map_path, od_eval)
        upload_if_needed(output_path, output_uri)


@click.command()
@click.argument('image_uris', nargs=-1)
@click.argument('label_map_uri')
@click.argument('ground_truth_uri')
@click.argument('predictions_uri')
@click.argument('output_uri')
@click.option('--save-temp', is_flag=True)
def eval_predictions(image_uris, label_map_uri, ground_truth_uri,
                     predictions_uri, output_uri, save_temp):
    """Evaluate predictions against ground truth for a single predictions file.

    Args:
        ground_truth_uri: GeoJSON file with ground truth bounding boxes
        predictions_uri: GeoJSON file with predicted bounding boxes
        output_uri: JSON file with metrics
    """
    _eval_predictions(image_uris, label_map_uri, ground_truth_uri,
                      predictions_uri, output_uri, save_temp)


if __name__ == '__main__':
    eval_predictions()
