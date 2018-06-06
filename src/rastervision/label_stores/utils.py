import copy


def add_classes_to_geojson(geojson, class_map):
    """Add missing class_names and class_ids from label GeoJSON."""
    geojson = copy.deepcopy(geojson)
    features = geojson['features']

    for feature in features:
        properties = feature.get('properties', {})
        if 'class_id' not in properties:
            if 'class_name' in properties:
                properties['class_id'] = \
                    class_map.get_by_name(properties['class_name']).id
            elif 'label' in properties:
                # label is considered a synonym of class_name for now in order
                # to interface with Raster Foundry.
                properties['class_id'] = \
                    class_map.get_by_name(properties['label']).id
                properties['class_name'] = properties['label']
            else:
                # if no class_id, class_name, or label, then just assume
                # everything corresponds to class_id = 1.
                class_id = 1
                class_name = class_map.get_by_id(class_id).name
                properties['class_id'] = class_id
                properties['class_name'] = class_name

        feature['properties'] = properties

    return geojson
