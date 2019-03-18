import json

from rastervision.data.vector_source.vector_source import VectorSource
from rastervision.utils.files import file_to_str


class GeoJSONVectorSource(VectorSource):
    def __init__(self,
                 uri,
                 crs_transformer,
                 line_bufs=None,
                 point_bufs=None,
                 class_inf_opts=None):
        """Constructor.

        Args:
            uri: (str) uri of GeoJSON file
            class_inf_opts: ClassInferenceOptions
        """
        self.uri = uri
        super().__init__(crs_transformer, line_bufs, point_bufs,
                         class_inf_opts)

    def _get_geojson(self):
        geojson = json.loads(file_to_str(self.uri))
        return self.class_inference.transform_geojson(geojson)
