import json
import argparse

from shapely.geometry import shape


def filter_geojson(mask_path, input_path, output_path):
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


def parse_args():
    description = """
        Filter out polygons that are contained in a mask multipolygon. Input
        and output is GeoJSON.
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--mask-path', help='GeoJSON file with mask')
    parser.add_argument('--input-path',
                        help='GeoJSON file with polygons to filter')
    parser.add_argument('--output-path')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    filter_geojson(
        args.mask_path, args.input_path, args.output_path)
