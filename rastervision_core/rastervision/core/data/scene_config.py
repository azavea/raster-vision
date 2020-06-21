from typing import Optional, List

from shapely.geometry import shape

from rastervision.pipeline.config import Config, register_config, Field
from rastervision.core.data.raster_source import RasterSourceConfig
from rastervision.core.data.label_source import LabelSourceConfig
from rastervision.core.data.label_store import LabelStoreConfig
from rastervision.core.data.scene import Scene
from rastervision.core.data.vector_source import GeoJSONVectorSourceConfig


@register_config('scene')
class SceneConfig(Config):
    """Config for a Scene which comprises the raster data and labels for an AOI."""
    id: str
    raster_source: RasterSourceConfig
    label_source: LabelSourceConfig
    label_store: Optional[LabelStoreConfig] = None
    aoi_uris: Optional[List[str]] = Field(
        None,
        description=
        ('List of URIs of GeoJSON files that define the AOIs for the scene. Each polygon'
         'defines an AOI which is a piece of the scene that is assumed to be fully '
         'labeled and usable for training or validation.'))

    def build(self, class_config, tmp_dir, use_transformers=True):
        raster_source = self.raster_source.build(
            tmp_dir, use_transformers=use_transformers)
        crs_transformer = raster_source.get_crs_transformer()
        extent = raster_source.get_extent()

        label_source = (self.label_source.build(class_config, crs_transformer,
                                                extent, tmp_dir)
                        if self.label_source is not None else None)
        label_store = (self.label_store.build(class_config, crs_transformer,
                                              extent, tmp_dir)
                       if self.label_store is not None else None)

        aoi_polygons = None
        if self.aoi_uris is not None:
            aoi_polygons = []
            for uri in self.aoi_uris:
                # Set default class id to 0 to avoid deleting features. If it was
                # set to None, they would all be deleted.
                aoi_geojson = GeoJSONVectorSourceConfig(
                    uri=uri, default_class_id=0, ignore_crs_field=True).build(
                        class_config, crs_transformer).get_geojson()
                for f in aoi_geojson['features']:
                    aoi_polygons.append(shape(f['geometry']))

        return Scene(
            self.id,
            raster_source,
            ground_truth_label_source=label_source,
            prediction_label_store=label_store,
            aoi_polygons=aoi_polygons)

    def update(self, pipeline=None):
        super().update()

        self.raster_source.update(pipeline=pipeline, scene=self)
        self.label_source.update(pipeline=pipeline, scene=self)
        if self.label_store is None and pipeline is not None:
            self.label_store = pipeline.get_default_label_store(scene=self)
        if self.label_store is not None:
            self.label_store.update(pipeline=pipeline, scene=self)
