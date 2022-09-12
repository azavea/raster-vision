from typing import TYPE_CHECKING, List

from rastervision.core.data.vector_source.vector_source import VectorSource
from rastervision.pipeline.file_system import download_if_needed, file_to_json

if TYPE_CHECKING:
    from rastervision.core.data import CRSTransformer, VectorTransformer


class GeoJSONVectorSource(VectorSource):
    def __init__(self,
                 uri: str,
                 crs_transformer: 'CRSTransformer',
                 vector_transformers: List['VectorTransformer'] = [],
                 ignore_crs_field: bool = False):
        self.uri = uri
        self.ignore_crs_field = ignore_crs_field
        super().__init__(
            crs_transformer, vector_transformers=vector_transformers)

    def _get_geojson(self):
        # download first so that it gets cached
        geojson = file_to_json(download_if_needed(self.uri))
        if not self.ignore_crs_field and 'crs' in geojson:
            raise Exception(
                f'The GeoJSON file at {self.uri} contains a CRS field which '
                'is not allowed by the current GeoJSON standard or by '
                'Raster Vision. All coordinates are expected to be in '
                'EPSG:4326 CRS. If the file uses EPSG:4326 (ie. lat/lng on '
                'the WGS84 reference ellipsoid) and you would like to ignore '
                'the CRS field, set ignore_crs_field=True in '
                'GeoJSONVectorSourceConfig.')

        return geojson
