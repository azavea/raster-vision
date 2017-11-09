from os.path import join
from shutil import move

import click

from rv.commands.make_predict_chips import _make_predict_chips
from rv.commands.predict_on_chips import _predict_on_chips
from rv.commands.aggregate_predictions import _aggregate_predictions
from rv.commands.transform_geojson import _transform_geojson
from rv.commands.settings import planet_channel_order, temp_root_dir
from rv.commands.utils import (
    download_if_needed, upload_if_needed, get_local_path, make_temp_dir,
    download_and_build_vrt)


def _predict(inference_graph_uri, label_map_uri, image_uris,
             agg_predictions_uri, agg_predictions_debug_uri=None,
             mask_uri=None, channel_order=planet_channel_order, chip_size=300,
             score_thresh=0.5, merge_thresh=0.05):
    temp_dir = join(temp_root_dir, 'predict')
    make_temp_dir(temp_dir)

    # Download input files if needed.
    inference_graph_path = download_if_needed(temp_dir, inference_graph_uri)
    label_map_path = download_if_needed(temp_dir, label_map_uri)
    mask_path = download_if_needed(temp_dir, mask_uri)
    image_path = download_and_build_vrt(temp_dir, image_uris)

    # Output files can't be downloaded since they don't exist yet, but
    # we need to figure out where to store them locally if the URI is remote.
    agg_predictions_path = get_local_path(temp_dir, agg_predictions_uri)
    agg_predictions_debug_path = get_local_path(
        temp_dir, agg_predictions_debug_uri)

    # Divide VRT into overlapping chips.
    chips_dir = join(temp_dir, 'chips')
    chips_info_path = join(temp_dir, 'chips_info.json')
    _make_predict_chips(
        image_path, chips_dir, chips_info_path, chip_size=chip_size,
        channel_order=channel_order)

    # Make prediction for each chip.
    predictions_path = join(temp_dir, 'predictions.json')
    predictions_debug_dir = join(temp_dir, 'predictions_debug')
    _predict_on_chips(inference_graph_path, label_map_path, chips_dir,
                      predictions_path,
                      predictions_debug_dir=predictions_debug_dir)

    # Aggregate predictions from local into global coordinate frame.
    _aggregate_predictions(image_path, chips_info_path, predictions_path,
                           label_map_path, agg_predictions_path,
                           agg_predictions_debug_path=agg_predictions_debug_path,  # noqa
                           channel_order=channel_order)

    # Filter out predictions.
    if mask_path is not None:
        unfiltered_predictions_path = join(
            temp_dir, 'unfiltered_predictions.geojson')
        move(agg_predictions_path, unfiltered_predictions_path)
        _transform_geojson(
            mask_path, unfiltered_predictions_path, agg_predictions_path)

    # Upload output files if the URIs are remote.
    upload_if_needed(agg_predictions_path, agg_predictions_uri)
    upload_if_needed(agg_predictions_debug_path, agg_predictions_debug_uri)


@click.command()
@click.argument('inference_graph_uri')
@click.argument('label_map_uri')
@click.argument('image_uris', nargs=-1)
@click.argument('agg_predictions_uri')
@click.option('--agg-predictions-debug-uri', default=None,
              help='URI for prediction debug plot')
@click.option('--mask-uri', default=None,
              help='URI for mask GeoJSON file to use as filter for detections')
@click.option('--channel-order', nargs=3, type=int,
              default=planet_channel_order, help='Index of RGB channels')
@click.option('--chip-size', default=300)
@click.option('--score-thresh', default=0.5,
              help='Score threshold of predictions to keep')
@click.option('--merge-thresh', default=0.05,
              help='IOU threshold for merging predictions')
def predict(inference_graph_uri, label_map_uri, image_uris,
            agg_predictions_uri, agg_predictions_debug_uri, mask_uri,
            channel_order, chip_size, score_thresh, merge_thresh):
    """High-level script for running object detection over geospatial imagery.

    Args:
        image_uris: List of URIs for TIFF files to run prediction on
        agg_predictions_uri: Output file with aggregated predictions
    """
    _predict(inference_graph_uri, label_map_uri, image_uris,
             agg_predictions_uri, agg_predictions_debug_uri, mask_uri,
             channel_order, chip_size, score_thresh, merge_thresh)


if __name__ == '__main__':
    predict()
