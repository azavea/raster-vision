from shapely import geometry


def aoi_json_to_shapely(geojson_dict, crs_transformer):
    """Load geojson as shapely polygon

    Returns list of shapely polygons for geojson uri or None if
    uri doesn't exist
    """
    aoi_shapely = aoi_geojson_to_shapely_polygons(geojson_dict,
                                                  crs_transformer)
    return aoi_shapely


def aoi_geojson_to_shapely_polygons(geojson_dict, crs_transformer):
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
    polygons = []

    for feature in features:
        json_polygons = []
        # Convert polygon to pixel coords.
        geom_type = feature['geometry']['type']
        coordinates = feature['geometry']['coordinates']
        if geom_type == 'MultiPolygon':
            for polygon in coordinates:
                shell = polygon[0]
                json_polygons.append(
                    [crs_transformer.map_to_pixel(p) for p in shell])
        elif geom_type == 'Polygon':
            shell = coordinates[0]
            json_polygons.append(
                [crs_transformer.map_to_pixel(p) for p in shell])
        else:
            raise Exception('Geometries of type {} are not supported in AOIs'
                            .format(geom_type))

        for json_polygon in json_polygons:
            polygon = geometry.Polygon([(p[0], p[1]) for p in json_polygon])
            # Trick to handle self-intersecting polygons which otherwise cause an
            # error.
            polygon = polygon.buffer(0)
            polygons.append(polygon)

    return polygons
