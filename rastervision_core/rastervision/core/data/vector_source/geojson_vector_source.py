from typing import TYPE_CHECKING
import logging

import geopandas as gpd

from rastervision.pipeline.file_system import download_if_needed
from rastervision.core.box import Box
from rastervision.core.data.vector_source.vector_source import VectorSource
from rastervision.core.data.utils import listify_uris, merge_geojsons

if TYPE_CHECKING:
    from rastervision.core.data import CRSTransformer, VectorTransformer

log = logging.getLogger(__name__)


class GeoJSONVectorSource(VectorSource):
    """A :class:`.VectorSource` for reading GeoJSON files."""

    def __init__(self,
                 uris: str | list[str],
                 crs_transformer: 'CRSTransformer',
                 vector_transformers: list['VectorTransformer'] = [],
                 bbox: Box | None = None):
        """Constructor.

        Args:
            uris: URI(s) of the GeoJSON file(s).
            crs_transformer: A ``CRSTransformer`` to convert between map and
                pixel coords. Normally this is obtained from a
                :class:`.RasterSource`.
            vector_transformers: ``VectorTransformers`` for transforming
                geometries. Defaults to ``[]``.
            bbox: User-specified crop of the extent. If ``None``, the full
                extent available in the source file is used.
        """
        self.uris = listify_uris(uris)
        super().__init__(
            crs_transformer,
            vector_transformers=vector_transformers,
            bbox=bbox)

    def _get_geojson(self) -> dict:
        geojsons = [self._get_geojson_single(uri) for uri in self.uris]
        geojson = merge_geojsons(geojsons)
        return geojson

    def _get_geojson_single(self, uri: str) -> dict:
        # download first so that it gets cached
        path = download_if_needed(uri)
        df: gpd.GeoDataFrame = gpd.read_file(path)
        df = df.to_crs('epsg:4326')
        geojson = df.__geo_interface__
        return geojson
