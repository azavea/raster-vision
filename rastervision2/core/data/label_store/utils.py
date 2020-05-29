def boxes_to_geojson(  # noqa
        boxes,  # noqa
        class_ids,
        crs_transformer,
        class_config,
        scores=None):
    """Convert boxes and associated data into a GeoJSON dict.

    Args:
        boxes: list of Box in pixel row/col format.
        class_ids: list of int (one for each box)
        crs_transformer: CRSTransformer used to convert pixel coords to map
            coords in the GeoJSON
        class_config: ClassConfig
        scores: optional list of score or scores.
                If floats (one for each box), property name will be "score".
                If lists of floats, property name will be "scores".

    Returns:
        dict in GeoJSON format
    """
    features = []
    for box_ind, box in enumerate(boxes):
        polygon = box.geojson_coordinates()
        polygon = [list(crs_transformer.pixel_to_map(p)) for p in polygon]

        class_id = int(class_ids[box_ind])
        class_name = class_config.get_name(class_id)

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            },
            'properties': {
                'class_id': class_id,
                'class_name': class_name
            }
        }

        if scores is not None:
            box_scores = scores[box_ind]

            if box_scores is not None:
                if type(box_scores) is list:
                    feature['properties']['scores'] = box_scores
                else:
                    feature['properties']['score'] = box_scores

        features.append(feature)

    return {'type': 'FeatureCollection', 'features': features}
