import copy

from rastervision.data.vector_source.label_maker.filter import create_filter


class ClassInferenceOptions():
    def __init__(self,
                 class_map=None,
                 class_id_to_filter=None,
                 default_class_id=1):
        self.class_map = class_map
        self.class_id_to_filter = class_id_to_filter
        self.default_class_id = default_class_id


class ClassInference():
    """Infers missing class_ids from GeoJSON features."""

    def __init__(self, options):
        """Constructor.

        Args:
            options: (ClassInferenceOptions)
        """
        self.options = options

        if self.options.class_id_to_filter is not None:
            self.class_id_to_filter = {}
            for class_id, filter_exp in self.options.class_id_to_filter.items(
            ):
                self.class_id_to_filter[class_id] = create_filter(filter_exp)

    def infer_class_id(self, feature):
        """Infer the class_id for a GeoJSON feature.

        Args:
            feature: (dict) GeoJSON feature

        Rules:
            1) If class_id is in feature['properties'], use it.
            2) If class_name or label are in feature['properties'] and in class_map,
                use corresponding class_id.
            3) If class_id_to_filter is set and filter is true when applied to feature,
                use corresponding class_id.
            4) Otherwise, return the default_class_id
        """
        class_id = feature.get('properties', {}).get('class_id')
        if class_id is not None:
            return class_id

        if self.options.class_map is not None:
            class_name = feature.get('properties', {}).get('class_name')
            if class_name in self.options.class_map.get_class_names():
                return self.options.class_map.get_by_name(class_name).id

            label = feature.get('properties', {}).get('label')
            if label in self.options.class_map.get_class_names():
                return self.options.class_map.get_by_name(label).id

        if self.options.class_id_to_filter is not None:
            for class_id, filter_fn in self.class_id_to_filter.items():
                if filter_fn(feature):
                    return class_id

        return self.options.default_class_id

    def transform_geojson(self, geojson):
        """Transform GeoJSON by appending class_ids and removing features with no class.

        For each feature in geojson, the class_id is inferred and is set into
        feature['properties']. If the class_id is None (because none of the rules apply
        and the default_class_id is None), the feature is dropped.
        """
        new_features = []
        for feature in geojson['features']:
            class_id = self.infer_class_id(feature)
            if class_id is not None:
                feature = copy.deepcopy(feature)
                properties = feature.get('properties', {})
                properties['class_id'] = class_id
                feature['properties'] = properties
                new_features.append(feature)
        new_geojson = {'type': 'FeatureCollection', 'features': new_features}
        return new_geojson
