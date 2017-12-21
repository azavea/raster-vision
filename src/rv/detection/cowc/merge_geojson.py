import os
import glob
import json

import click


@click.command()
@click.argument('input_dir')
@click.argument('output_path')
def merge_geojson(input_dir, output_path):
    input_paths = glob.glob(os.path.join(input_dir, '*.json'))
    features = []
    for input_path in input_paths:
        with open(input_path, 'r') as input_file:
            input_json = json.load(input_file)
            features.extend(input_json['features'])

    with open(input_path, 'r') as input_file:
        output_json = json.load(input_file)
        output_json['features'] = features
        with open(output_path, 'w') as output_file:
            json.dump(output_json, output_file, indent=4)


if __name__ == '__main__':
    merge_geojson()
