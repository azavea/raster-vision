# flake8: noqa

# Registry Keys
LABEL_STORE = 'LABEL_STORE'

# Use the same labels as for the source for these label stores.
from rastervision.data.label_source.api import (OBJECT_DETECTION_GEOJSON,
                                                CHIP_CLASSIFICATION_GEOJSON,
                                                SEMANTIC_SEGMENTATION_RASTER)

from rastervision.data.label_store.label_store_config import LabelStoreConfig
