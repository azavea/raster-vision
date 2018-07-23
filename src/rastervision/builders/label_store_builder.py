from rastervision.label_stores.object_detection_geojson_file import (
    ObjectDetectionGeoJSONFile)
from rastervision.label_stores.classification_geojson_file import (
    ClassificationGeoJSONFile)
from rastervision.label_stores.segmentation_raster_file import (
    SegmentationRasterFile)


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
        return SegmentationRasterFile(
            src=config.segmentation_raster_file.src,
            dst=config.segmentation_raster_file.dst,
            src_classes=config.segmentation_raster_file.src_classes,
            dst_classes=config.segmentation_raster_file.dst_classes)
        return None
    else:
        raise ValueError(
            'Not sure how to generate label store for type {}'
            .format(label_store_type))
