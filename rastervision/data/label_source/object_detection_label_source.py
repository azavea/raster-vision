import rastervision as rv
from rastervision.data.label import ObjectDetectionLabels
from rastervision.data.label_source import LabelSource


class ObjectDetectionLabelSource(LabelSource):
    def __init__(self, vector_source, crs_transformer, class_map, extent):
        """Constructor.

        Args:
            vector_source: (VectorSource or str)
            crs_transformer: CRSTransformer to convert from map coords in label
                in GeoJSON file to pixel coords.
            class_map: ClassMap used to infer class_ids from class_name
                (or label) field
            extent: Box used to filter the labels by extent
        """
        if isinstance(vector_source, str):
            provider = rv._registry.get_vector_source_default_provider(
                vector_source)
            vector_source = provider.construct(vector_source) \
                .create_source(
                    crs_transformer=crs_transformer, extent=extent, class_map=class_map)

        self.labels = ObjectDetectionLabels.from_geojson(
            vector_source.get_geojson(), extent=extent)

    def get_labels(self, window=None):
        if window is None:
            return self.labels

        return ObjectDetectionLabels.get_overlapping(self.labels, window)
