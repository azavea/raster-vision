from shapely import geometry


def json_to_shapely(geojson_dict, crs_transformer):
    """Load geojson as shapely polygon

    Returns list of shapely polygons for geojson uri or None if
    uri doesn't exist
    """
    aoi_shapely = geojson_to_shapely_polygons(geojson_dict, crs_transformer)
    return aoi_shapely


def geojson_to_shapely_polygons(geojson_dict, crs_transformer):
    """Get list of shapely polygons from geojson dict

    Args:
        geojson_dict: dict in GeoJSON format with class_id property for each
            polygon
        crs_transformer: CRSTransformer used to convert from map to pixel
            coords

    Returns:
        list of shapely.geometry.Polygon
    """
    if not geojson_dict:
        return None

    features = geojson_dict['features']
    json_polygons = []
    class_ids = []

    for feature in features:
        # Convert polygon to pixel coords.
        polygon = feature['geometry']['coordinates'][0]
        polygon = [crs_transformer.map_to_pixel(p) for p in polygon]
        json_polygons.append(polygon)

        properties = feature.get('properties', {})
        class_ids.append(properties.get('class_id', 1))

    # Convert polygons to shapely
    polygons = []
    for json_polygon, class_id in zip(json_polygons, class_ids):
        polygon = geometry.Polygon([(p[0], p[1]) for p in json_polygon])
        # Trick to handle self-intersecting polygons which otherwise cause an
        # error.
        polygon = polygon.buffer(0)
        polygon.class_id = class_id
        polygons.append(polygon)
    return polygons
