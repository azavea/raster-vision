from rastervision.core.scene import Scene
from rastervision.builders import label_store_builder, raster_source_builder
from rastervision.label_stores.utils import json_to_shapely


def build(config, class_map, predictions_readable=False):
    raster_source = None
    crs_transformer = None
    ground_truth_label_store = None
    prediction_label_store = None
    aoi_polygons = None

    raster_source = raster_source_builder.build(config.raster_source)
    extent = raster_source.get_extent()
    crs_transformer = raster_source.get_crs_transformer()

    if config.HasField('ground_truth_label_store'):
        ground_truth_label_store = label_store_builder.build(
            config.ground_truth_label_store,
            crs_transformer,
            extent,
            class_map,
            readable=True,
            writable=False)

    if config.HasField('prediction_label_store'):
        prediction_label_store = label_store_builder.build(
            config.prediction_label_store,
            crs_transformer,
            extent,
            class_map,
            readable=predictions_readable,
            writable=True)

    if config.HasField('aoi_uri'):
        aoi_polygons = json_to_shapely(config.aoi_uri, crs_transformer)

    return Scene(
        id=config.id,
        raster_source=raster_source,
        ground_truth_label_store=ground_truth_label_store,
        prediction_label_store=prediction_label_store,
        aoi_polygons=aoi_polygons)
