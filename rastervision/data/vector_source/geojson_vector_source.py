import json

from rastervision.data.vector_source.vector_source import VectorSource
from rastervision.utils.files import file_to_str


class GeoJSONVectorSource(VectorSource):
    def __init__(self, uri, class_inf_opts=None):
        """Constructor.

        Args:
            uri: (str) uri of GeoJSON file
            class_inf_opts: ClassInferenceOptions
        """
        self.uri = uri
        super().__init__(class_inf_opts)

    def _get_geojson(self):
        return json.loads(file_to_str(self.uri))
