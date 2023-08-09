from typing import TYPE_CHECKING, List, Optional, Union
import logging

from rastervision.pipeline.file_system import download_if_needed, file_to_json
from rastervision.core.box import Box
from rastervision.core.data.vector_source.vector_source import VectorSource
from rastervision.core.data.utils import listify_uris, merge_geojsons

if TYPE_CHECKING:
    from rastervision.core.data import CRSTransformer, VectorTransformer

log = logging.getLogger(__name__)


class GeoJSONVectorSource(VectorSource):
    """A :class:`.VectorSource` for reading GeoJSON files."""

    def __init__(self,
                 uris: Union[str, List[str]],
                 crs_transformer: 'CRSTransformer',
                 vector_transformers: List['VectorTransformer'] = [],
                 ignore_crs_field: bool = False,
                 bbox: Optional[Box] = None):
        """Constructor.

        Args:
            uris (Union[str, List[str]]): URI(s) of the GeoJSON file(s).
            crs_transformer: A ``CRSTransformer`` to convert
                between map and pixel coords. Normally this is obtained from a
                :class:`.RasterSource`.
            vector_transformers: ``VectorTransformers`` for transforming
                geometries. Defaults to ``[]``.
            ignore_crs_field (bool): Ignore the CRS specified in the file and
                assume WGS84 (EPSG:4326) coords. Only WGS84 is supported
                currently. If False, and the file contains a CRS, will throw an
                exception on read. Defaults to False.
            bbox (Optional[Box]): User-specified crop of the extent. If None,
                the full extent available in the source file is used.
        """
        self.uris = listify_uris(uris)
        self.ignore_crs_field = ignore_crs_field
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
        geojson = file_to_json(download_if_needed(uri))
        if 'crs' in geojson:
            if not self.ignore_crs_field:
                raise NotImplementedError(
                    f'The GeoJSON file at {uri} contains a CRS field which '
                    'is not allowed by the current GeoJSON standard or by '
                    'Raster Vision. All coordinates are expected to be in '
                    'EPSG:4326 CRS. If the file uses EPSG:4326 (ie. lat/lng '
                    'on the WGS84 reference ellipsoid) and you would like to '
                    'ignore the CRS field, set ignore_crs_field=True.')
            else:
                crs = geojson['crs']
                log.info(f'Ignoring CRS ({crs}) specified in {uri} '
                         'and assuming EPSG:4326 instead.')
                # Delete the CRS field to avoid discrepancies in case the
                # geojson is passed to code that *does* respect the CRS field
                # (e.g. geopandas).
                del geojson['crs']
                # Also delete any "crs" keys in features' properties.
                for f in geojson.get('features', []):
                    try:
                        del f['properties']['crs']
                    except KeyError:
                        pass
        return geojson
