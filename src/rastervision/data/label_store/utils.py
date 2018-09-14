def boxes_to_geojson(boxes, class_ids, crs_transformer, class_map,
                     scores=None):
    """Convert boxes and associated data into a GeoJSON dict.

    Args:
        boxes: list of Box in pixel row/col format.
        class_ids: list of int (one for each box)
        crs_transformer: CRSTransformer used to convert pixel coords to map
            coords in the GeoJSON
        class_map: ClassMap used to infer class_name from class_id
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
        class_name = class_map.get_by_id(class_id).name

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


def classification_labels_to_geojson(labels, crs_transformer, class_map):
    """Return a geojson dict from classification labels.
    """
    boxes = labels.get_cells()
    class_ids = labels.get_class_ids()
    scores = list(labels.get_scores())

    return boxes_to_geojson(boxes, class_ids, crs_transformer, class_map,
                            scores)
