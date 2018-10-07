from abc import (ABC, abstractmethod)
import os

import rastervision as rv


class LabelStoreDefaultProvider(ABC):
    @staticmethod
    @abstractmethod
    def is_default_for(task_type):
        """Returns True if this label store is the default for this tasks_type"""
        pass

    @staticmethod
    @abstractmethod
    def handles(task_type, s):
        """Returns True of this provider is a default for this task_type and string"""
        pass

    @abstractmethod
    def construct(s=None):
        """Construts a default LabelStore based on the string.
        """
        pass


class ObjectDetectionGeoJSONStoreDefaultProvider(LabelStoreDefaultProvider):
    @staticmethod
    def is_default_for(task_type):
        return task_type == rv.OBJECT_DETECTION

    @staticmethod
    def handles(task_type, uri):
        if task_type == rv.OBJECT_DETECTION:
            ext = os.path.splitext(uri)[1]
            return ext.lower() in ['.json', '.geojson']
        return False

    @staticmethod
    def construct(uri=None):
        b = rv.LabelStoreConfig.builder(rv.OBJECT_DETECTION_GEOJSON)
        if uri:
            b = b.with_uri(uri)

        return b.build()


class ChipClassificationGeoJSONStoreDefaultProvider(LabelStoreDefaultProvider):
    @staticmethod
    def is_default_for(task_type):
        return task_type == rv.CHIP_CLASSIFICATION

    @staticmethod
    def handles(task_type, uri):
        if task_type == rv.CHIP_CLASSIFICATION:
            ext = os.path.splitext(uri)[1]
            return ext.lower() in ['.json', '.geojson']
        return False

    @staticmethod
    def construct(uri=None):
        b = rv.LabelStoreConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON)
        if uri:
            b = b.with_uri(uri)
        return b.build()


class SemanticSegmentationRasterStoreDefaultProvider(
        LabelStoreDefaultProvider):
    @staticmethod
    def is_default_for(task_type):
        return task_type == rv.SEMANTIC_SEGMENTATION

    @staticmethod
    def handles(task_type, uri):
        if task_type == rv.SEMANTIC_SEGMENTATION:
            ext = os.path.splitext(uri)[1]
            return ext.lower() in ['.tiff', '.tif']
        return False

    @staticmethod
    def construct(uri=None):
        b = rv.LabelStoreConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER)
        if uri:
            b = b.with_uri(uri)
        return b.build()
