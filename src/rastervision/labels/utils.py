def boxes_to_geojson(boxes, class_ids, crs_transformer, class_map, scores=None):
    features = []
    for box_ind, box in enumerate(boxes):
        polygon = box.geojson_coordinates()
        polygon = [crs_transformer.pixel_to_web(p) for p in polygon]

        class_id = class_ids[box_ind]
        class_name = class_map.get_by_id(class_id).name
        score = 0.0
        if scores is not None:
            score = scores[box_ind]

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            },
            'properties': {
                'class_id': class_id,
                'class_name': class_name,
                'score': score
            }
        }
        features.append(feature)

    return {
        'type': 'FeatureCollection',
        'features': features
    }
