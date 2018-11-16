from abc import ABC, abstractmethod

from rastervision.data.vector_source.class_inference import (
    ClassInference, ClassInferenceOptions)


class VectorSource(ABC):
    """A source of vector data.

    Uses GeoJSON as its internal representation of vector data.
    """

    def __init__(self, class_inf_opts=None):
        """Constructor.

        Args:
            class_inf_opts: (ClassInferenceOptions)
        """
        if class_inf_opts is None:
            class_inf_opts = ClassInferenceOptions()
        self.class_inference = ClassInference(class_inf_opts)
        self.geojson = None

    def get_geojson(self):
        if self.geojson is None:
            self.geojson = self.class_inference.transform_geojson(
                self._get_geojson())
        return self.geojson

    @abstractmethod
    def _get_geojson(self):
        pass
