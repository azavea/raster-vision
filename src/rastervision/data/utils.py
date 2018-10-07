import shapely


def geojson_to_shapes(geojson, crs_transformer):
    """Convert GeoJSON into list of shapely.geometry shape.

    Args:
        geojson: dict in GeoJSON format with class_id property for each
            feature (class_id defaults to 1 if missing)
        crs_transformer: CRSTransformer used to convert from map to pixel
            coords

    Returns:
        List of (shapely.geometry, class_id) tuples
    """
    features = geojson['features']
    shapes = []

    for feature in features:
        properties = feature.get('properties', {})
        class_id = properties.get('class_id', 1)
        geom_type = feature['geometry']['type']
        coordinates = feature['geometry']['coordinates']

        if geom_type == 'MultiPolygon':
            for polygon in coordinates:
                shell = polygon[0]
                shape = [crs_transformer.map_to_pixel(p) for p in shell]
                # Trick to handle self-intersecting polygons using buffer(0)
                shape = shapely.geometry.Polygon(shape).buffer(0)
                shapes.append((shape, class_id))
        elif geom_type == 'Polygon':
            shell = coordinates[0]
            shape = [crs_transformer.map_to_pixel(p) for p in shell]
            # Trick to handle self-intersecting polygons using buffer(0)
            shape = shapely.geometry.Polygon(shape).buffer(0)
            shapes.append((shape, class_id))
        elif geom_type == 'LineString':
            shape = [crs_transformer.map_to_pixel(p) for p in coordinates]
            shape = shapely.geometry.LineString(shape)
            shapes.append((shape, class_id))
        else:
            # TODO: logging warning that this type can't be parsed.
            pass

    return shapes


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
