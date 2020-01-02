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


class ObjectDetectionLabelSourceDefaultProvider(LabelSourceDefaultProvider):
    @staticmethod
    def handles(task_type, uri):
        if task_type == rv.OBJECT_DETECTION:
            ext = os.path.splitext(uri)[1]
            return ext.lower() in ['.geojson', '.json', '.mbtiles']

    @staticmethod
    def construct(uri):
        return rv.LabelSourceConfig.builder(rv.OBJECT_DETECTION) \
                                   .with_uri(uri) \
                                   .build()


class ChipClassificationLabelSourceDefaultProvider(LabelSourceDefaultProvider):
    @staticmethod
    def handles(task_type, uri):
        if task_type == rv.CHIP_CLASSIFICATION:
            ext = os.path.splitext(uri)[1]
            return ext.lower() in ['.geojson', '.json', '.mbtiles']

    @staticmethod
    def construct(uri):
        return rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION) \
                                   .with_uri(uri) \
                                   .build()


class SemanticSegmentationLabelSourceDefaultProvider(
        LabelSourceDefaultProvider):
    @staticmethod
    def handles(task_type, uri):
        if task_type == rv.SEMANTIC_SEGMENTATION:
            ext = os.path.splitext(uri)[1]
            return ext.lower() in ['.tif', '.tiff']
        return False

    @staticmethod
    def construct(uri):
        return rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                                   .with_raster_source(uri) \
                                   .build()
