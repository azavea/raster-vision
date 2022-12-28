from typing import TYPE_CHECKING, List

from rastervision.core.data.vector_source.vector_source import VectorSource
from rastervision.pipeline.file_system import download_if_needed, file_to_json

if TYPE_CHECKING:
    from rastervision.core.data import CRSTransformer, VectorTransformer


class GeoJSONVectorSource(VectorSource):
    """A :class:`.VectorSource` for reading GeoJSON files."""

    def __init__(self,
                 uri: str,
                 crs_transformer: 'CRSTransformer',
                 vector_transformers: List['VectorTransformer'] = [],
                 ignore_crs_field: bool = False):
        """Constructor.

        Args:
            uri (str): URI of the GeoJSON file.
            crs_transformer: A ``CRSTransformer`` to convert
                between map and pixel coords. Normally this is obtained from a
                :class:`.RasterSource`.
            vector_transformers: ``VectorTransformers`` for transforming
                geometries. Defaults to ``[]``.
            ignore_crs_field (bool): Ignore the CRS specified in the file and
                assume WGS84 (EPSG:4326) coords. Only WGS84 is supported
                currently. If False, and the file contains a CRS, will throw an
                exception on read. Defaults to False.
        """
        self.uri = uri
        self.ignore_crs_field = ignore_crs_field
        super().__init__(
            crs_transformer, vector_transformers=vector_transformers)

    def _get_geojson(self):
        # download first so that it gets cached
        geojson = file_to_json(download_if_needed(self.uri))
        if not self.ignore_crs_field and 'crs' in geojson:
            raise NotImplementedError(
                f'The GeoJSON file at {self.uri} contains a CRS field which '
                'is not allowed by the current GeoJSON standard or by '
                'Raster Vision. All coordinates are expected to be in '
                'EPSG:4326 CRS. If the file uses EPSG:4326 (ie. lat/lng on '
                'the WGS84 reference ellipsoid) and you would like to ignore '
                'the CRS field, set ignore_crs_field=True in '
                'GeoJSONVectorSourceConfig.')

        return geojson
