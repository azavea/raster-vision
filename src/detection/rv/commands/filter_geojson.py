import json

import click
from shapely.geometry import shape


def _filter_geojson(mask_path, input_path, output_path):
    with open(input_path) as input_file:
        input_dict = json.load(input_file)

    with open(mask_path) as mask_file:
        mask_dict = json.load(mask_file)
        mask_geom = shape(mask_dict['features'][0]['geometry'])

    features = input_dict['features']
    filtered_features = []
    for feature in features:
        geom = shape(feature['geometry'])

        if not mask_geom.contains(geom):
            filtered_features.append(feature)

    output_dict = dict(input_dict)
    output_dict['features'] = filtered_features

    with open(output_path, 'w') as output_file:
        json.dump(output_dict, output_file, indent=4)


@click.command()
@click.argument('mask_path')
@click.argument('input_path')
@click.argument('output_path')
def filter_geojson(mask_path, input_path, output_path):
    """Filter out polygons that are contained in a mask multipolygon.

    Args:
        mask_path: GeoJSON file containing mask multipolygon
    """
    _filter_geojson(mask_path, input_path, output_path)


if __name__ == '__main__':
    filter_geojson()
