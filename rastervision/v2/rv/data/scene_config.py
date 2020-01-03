from typing import Optional, List

from rastervision.v2.core import Config
from rastervision.v2.rv.data.raster_source import RasterSourceConfig
from rastervision.v2.rv.data.label_source import LabelSourceConfig
from rastervision.v2.rv.data.label_store import LabelStoreConfig


class SceneConfig(Config):
    id: str
    raster_source: RasterSourceConfig
    label_source: LabelSourceConfig
    label_store: Optional[LabelStoreConfig] = None
    aoi_uris: Optional[List[str]] = None
