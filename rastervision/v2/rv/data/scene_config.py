from typing import Optional, List

from shapely.geometry import shape

from rastervision.v2.core.config import Config, register_config
from rastervision.v2.rv.data.raster_source import RasterSourceConfig
from rastervision.v2.rv.data.label_source import LabelSourceConfig
from rastervision.v2.rv.data.label_store import LabelStoreConfig
from rastervision.v2.rv.data.scene import Scene
from rastervision.v2.rv.data.vector_source import GeoJSONVectorSource

@register_config('scene')
class SceneConfig(Config):
    id: str
    raster_source: RasterSourceConfig
    label_source: LabelSourceConfig
    label_store: Optional[LabelStoreConfig] = None
    aoi_uris: Optional[List[str]] = None

    def build(self):
        raster_source = self.raster_source.build()
        extent = raster_source.get_extent()
        crs_transformer = raster_source.get_crs_transformer()

        label_source = self.label_source.build()
        label_store = self.label_store.build()
        aoi_polygons = None

        if self.aoi_uris:
            aoi_polygons = []
            for uri in self.aoi_uris:
                aoi_geojson = GeoJSONVectorSource(
                    uri, crs_transformer).get_geojson()
                for f in aoi_geojson['features']:
                    aoi_polygons.append(shape(f['geometry']))

        return Scene(
            self.id, raster_source, ground_truth_label_source=label_source,
            prediction_label_store=label_store, aoi_polygons=aoi_polygons)
