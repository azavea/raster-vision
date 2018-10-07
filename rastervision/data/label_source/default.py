from abc import (ABC, abstractmethod)
import os

import rastervision as rv


class LabelSourceDefaultProvider(ABC):
    @staticmethod
    @abstractmethod
    def handles(task_type, s):
        """Returns True of this provider is a default for this task_type and string"""
        pass

    @abstractmethod
    def construct(s):
        """Construts a default LabelSource based on the string.
        """
        pass


class ObjectDetectionGeoJSONSourceDefaultProvider(LabelSourceDefaultProvider):
    @staticmethod
    def handles(task_type, uri):
        if task_type == rv.OBJECT_DETECTION:
            ext = os.path.splitext(uri)[1]
            return ext.lower() in ['.json', '.geojson']
        return False

    @staticmethod
    def construct(uri):
        return rv.LabelSourceConfig.builder(rv.OBJECT_DETECTION_GEOJSON) \
                                   .with_uri(uri) \
                                   .build()


class ChipClassificationGeoJSONSourceDefaultProvider(
        LabelSourceDefaultProvider):
    @staticmethod
    def handles(task_type, uri):
        if task_type == rv.CHIP_CLASSIFICATION:
            ext = os.path.splitext(uri)[1]
            return ext.lower() in ['.json', '.geojson']
        return False

    @staticmethod
    def construct(uri):
        return rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                                   .with_uri(uri) \
                                   .build()


class SemanticSegmentationRasterSourceDefaultProvider(
        LabelSourceDefaultProvider):
    @staticmethod
    def handles(task_type, uri):
        if task_type == rv.SEMANTIC_SEGMENTATION:
            ext = os.path.splitext(uri)[1]
            return ext.lower() in ['.tif', '.tiff']
        return False

    @staticmethod
    def construct(uri):
        return rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
                                   .with_raster_source(uri) \
                                   .build()
