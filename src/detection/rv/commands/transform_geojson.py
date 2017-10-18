import json
from functools import partial

import click
import pyproj
from shapely.geometry import shape
import shapely.ops as ops
from shapely.geometry.polygon import Polygon


def get_area(feature):
    polygon = Polygon(feature['geometry']['coordinates'][0])
    # https://gis.stackexchange.com/questions/127607/area-in-km-from-polygon-of-coordinates  # noqa
    return ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat1=polygon.bounds[1],
                lat2=polygon.bounds[3])),
        polygon).area


def _transform_geojson(input_path, output_path, mask_path=None, min_area=None,
                       single_label=None):
    with open(input_path) as input_file:
        input_dict = json.load(input_file)

    mask_geom = None
    if mask_path is not None:
        with open(mask_path) as mask_file:
            mask_dict = json.load(mask_file)
            mask_geom = shape(mask_dict['features'][0]['geometry'])

    features = input_dict['features']
    filtered_features = []
    for feature in features:
        geom = shape(feature['geometry'])
        properties = feature.get('properties')
        if single_label is not None:
            if properties is not None:
                properties['label'] = single_label

        is_contained = mask_geom is None or mask_geom.contains(geom)
        area = get_area(feature)
        is_big_enough = area >= min_area
        if is_contained and is_big_enough:
            filtered_features.append(feature)

    output_dict = dict(input_dict)
    output_dict['features'] = filtered_features

    with open(output_path, 'w') as output_file:
        json.dump(output_dict, output_file, indent=4)


@click.command()
@click.argument('input_path')
@click.argument('output_path')
@click.option('--mask-path', help='GeoJSON file containing mask ' +
              'multipolygon. The output only contains features that are ' +
              'contained within the mask.')
@click.option('--min-area', default=0.0, help='The minimum allowed area of ' +
              'an object in meters^2')
@click.option('--single-label', help='The label to convert all labels to')
def transform_geojson(input_path, output_path, mask_path, min_area,
                      single_label):
    """Transform GeoJSON file.

    This generates a new GeoJSON based on an input GeoJSON file by filtering
    and transforming it in various ways controlled by the options.
    """
    _transform_geojson(input_path, output_path, mask_path, min_area,
                       single_label)


if __name__ == '__main__':
    transform_geojson()
