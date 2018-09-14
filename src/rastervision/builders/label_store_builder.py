from rastervision.label_stores.object_detection_geojson_file import (
    ObjectDetectionGeoJSONFile)
from rastervision.label_stores.classification_geojson_file import (
    ClassificationGeoJSONFile)
from rastervision.label_stores.segmentation_raster_file import (
    SegmentationInputRasterFile, SegmentationOutputRasterFile)


# TODO: Make this take a task config instead of a class map?
def build(config,
          crs_transformer,
          extent,
          class_map,
          readable=True,
          writable=False):

    label_store_type = config.WhichOneof('label_store_type')
    if label_store_type == 'object_detection_geojson_file':
        return ObjectDetectionGeoJSONFile(
            config.object_detection_geojson_file.uri,
            crs_transformer,
            class_map,
            extent=extent,
            readable=readable,
            writable=writable)
    elif label_store_type == 'classification_geojson_file':
        return ClassificationGeoJSONFile(
            config.classification_geojson_file.uri,
            crs_transformer,
            config.classification_geojson_file.options,
            class_map,
            extent,
            readable=readable,
            writable=writable)
    elif label_store_type == 'segmentation_raster_file':
        srf = config.segmentation_raster_file
        if not str(srf.source) == '':
            return SegmentationInputRasterFile(
                source=srf.source, raster_class_map=srf.raster_class_map)
        else:
            return SegmentationOutputRasterFile(
                sink=srf.sink, class_map=class_map)
    else:
        raise ValueError('Not sure how to generate label store for type {}'
                         .format(label_store_type))
