import tempfile
import zipfile
import os
import copy

from rastervision.utils.files import (download_if_needed, load_json_config,
                                      make_dir, save_json_config,
                                      get_local_path, upload_if_needed)
from rastervision.protos.predict_pb2 import PredictConfig
from rastervision.core.raster_stats import RasterStats
from rastervision.raster_sources.geotiff_files import GeoTiffFiles
from rastervision.core.raster_transformer import RasterTransformer

config_fn = 'predict-config.json'
model_fn = 'model'
stats_fn = 'stats.json'


def save_predict_package(predict_config):
    """Save a zip file with stuff needed to make predictions.

    Save a zip file containing model (model file), stats.json (output of
        compute_raster_stats command), and predict-config.json
        (predict command config file). This can be used by the predict_package
        command to make predictions on new images using the same configuration
        as in predict_config.

    Args:
        predict_config: (PredictConfig) with the desired location of the
            package in predict_config.options.prediction_package_uri
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        package_uri = predict_config.options.prediction_package_uri
        config_path = os.path.join(temp_dir, 'predict.json')
        save_json_config(predict_config, config_path)

        model_path = download_if_needed(predict_config.options.model_uri,
                                        temp_dir)
        stats_path = download_if_needed(
            predict_config.scenes[0]
            .raster_source.raster_transformer.stats_uri, temp_dir)

        package_path = get_local_path(package_uri, temp_dir)
        make_dir(package_path, use_dirname=True)
        with zipfile.ZipFile(package_path, 'w') as package_zip:
            package_zip.write(config_path, arcname=config_fn)
            package_zip.write(model_path, arcname=model_fn)
            package_zip.write(stats_path, arcname=stats_fn)
        upload_if_needed(package_path, package_uri)


def make_scene_config(scene_template,
                      stats_path,
                      labels_uri,
                      image_uris,
                      channel_order=None):
    """Create a scene config for making predictions from a template.

    Args:
        scene_template: rastervision.protos.scene_pb2.Scene
        stats_path: path to raster stats file
            (output of compute_raster_stats command)
        labels_uri: URI of labels file to write predictions to
        image_uris: list of URIS of image files to make predictions on

    Returns:
        (rastervision.protos.scene_pb2.Scene) copy of scene_template with
            the other arguments of this function injected into it
    """
    scene = copy.deepcopy(scene_template)
    scene.ground_truth_label_store.Clear()
    label_store = scene.prediction_label_store
    if label_store.HasField('object_detection_geojson_file'):
        label_store.object_detection_geojson_file.uri = \
            labels_uri
    elif label_store.HasField('classification_geojson_file'):
        label_store.classification_geojson_file.uri = \
            labels_uri

    del scene.raster_source.geotiff_files.uris[:]
    scene.raster_source.geotiff_files.uris.extend(image_uris)
    scene.raster_source.raster_transformer.stats_uri = stats_path

    if channel_order is not None:
        del scene.raster_source.raster_transformer.channel_order[:]
        scene.raster_source.raster_transformer.channel_order.extend(
            channel_order)

    return scene


def update_stats_file(image_uris, stats_path):
    """Write a raster stats file for a set of images.

    Args:
        image_uris: list of URIs of GeoTIFF files
        stats_path: path to write raster stats to
    """
    raster_transformer = RasterTransformer()
    raster_sources = [GeoTiffFiles(raster_transformer, image_uris)]

    stats = RasterStats()
    stats.compute(raster_sources)
    stats.save(stats_path)


def load_predict_package(predict_package_uri,
                         temp_dir,
                         labels_uri,
                         image_uris,
                         update_stats=False,
                         channel_order=None):
    """Load a prediction package to make predictions on new images.

    Downloads prediction package, unzips it, and returns a predict_config for
    running the predict command on new images.

    Args:
        predict_package_uri: URI of prediction package
            (see save_predict_package for details)
        temp_dir: path to directory to download model and stats files to
        labels_uri: URI of labels file to write predictions to
        image_uris: list of URIS of image files to make predictions on
        update_stats: if True, compute raster stats for image_uris and update
            raster stats file
        channel_order: if not None, list of channel indices to use

    Returns:
        (rastervision.protos.predict_pb2.PredictConfig)
    """
    # Download and extract package.
    package_zip_path = download_if_needed(predict_package_uri, temp_dir)
    package_dir = os.path.join(temp_dir, 'package')
    make_dir(package_dir)
    with zipfile.ZipFile(package_zip_path, 'r') as package_zip:
        package_zip.extractall(path=package_dir)

    config_path = os.path.join(package_dir, config_fn)
    stats_path = os.path.join(package_dir, stats_fn)
    model_path = os.path.join(package_dir, model_fn)

    # Update predict config.
    config = load_json_config(config_path, PredictConfig())
    config.options.debug = False
    config.options.debug_uri = ''
    config.options.prediction_package_uri = ''
    config.options.model_uri = model_path

    # Update scene config.
    scene_template = config.scenes[0]
    scene_config = make_scene_config(
        scene_template,
        stats_path,
        labels_uri,
        image_uris,
        channel_order=channel_order)
    del config.scenes[:]
    config.scenes.extend([scene_config])

    # Update stats file.
    if update_stats:
        update_stats_file(image_uris, stats_path)

    return config
