import copy
import json

from shapely.geometry import shape

from rastervision.utils.files import file_to_str


def boxes_to_geojson(boxes, class_ids, crs_transformer, class_map,
                     scores=None):
    """Convert boxes and associated data into a GeoJSON dict.

    Args:
        boxes: list of Box in pixel row/col format.
        class_ids: list of int (one for each box)
        crs_transformer: CRSTransformer used to convert pixel coords to map
            coords in the GeoJSON
        class_map: ClassMap used to infer class_name from class_id
        scores: optional list of floats (one for each box)


    Returns:
        dict in GeoJSON format
    """
    features = []
    for box_ind, box in enumerate(boxes):
        polygon = box.geojson_coordinates()
        polygon = [list(crs_transformer.pixel_to_map(p)) for p in polygon]

        class_id = int(class_ids[box_ind])
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

    return {'type': 'FeatureCollection', 'features': features}


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


def load_label_store_json(uri, readable):
    """Load JSON for LabelStore.

    Returns JSON for uri or None if it is not readable.
    """
    if not readable:
        return None

    return json.loads(file_to_str(uri))


def json_to_shapely(uri, crs_transformer):
    """Load geojson as shapely polygon 

    Returns list of shapely polygons for geojson uri or None if uri doesn't exist
    """
    if not uri:
        return None

    aoi_geojson = json.loads(file_to_str(uri))["features"]
    aoi_shapely = []
    for feature in aoi_geojson:
        coordinates = feature['geometry']['coordinates'][0]
        pixel_coordinates = []
        for c in coordinates:
            pixel_coordinate = list(crs_transformer.map_to_pixel((c[0], c[1])))
            pixel_coordinates.append(pixel_coordinate)
        feature['geometry']['coordinates'][0] = pixel_coordinates
        aoi_shapely.append(shape(feature["geometry"]))
    return aoi_shapely
