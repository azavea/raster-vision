from abc import (ABC, abstractmethod)
import os

import rastervision as rv


class DefaultLabelSourceProvider(ABC):
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


class DefaultObjectDetectionGeoJSONSourceProvider(DefaultLabelSourceProvider):
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


class DefaultChipClassificationGeoJSONSourceProvider(
        DefaultLabelSourceProvider):
    @staticmethod
    def handles(task_type, uri):
        if task_type == rv.CHIP_CLASSIFICATION:
            ext = os.path.splitext(uri)[1]
            return ext.lower() in ['.json', '.geojson']
        return False

    @staticmethod
    def construct(uri):
        return rv.RasterSourceConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                                    .with_uri(uri) \
                                    .build()
