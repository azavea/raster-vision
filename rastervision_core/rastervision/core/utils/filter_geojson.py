import json
import copy

import click

from rastervision.pipeline.file_system import file_to_str, str_to_file


@click.command()
@click.argument('labels_uri')
@click.argument('output_uri')
@click.argument('class_names', nargs=-1)
def filter_geojson(labels_uri, output_uri, class_names):
    """Remove features that aren't in class_names and remove class_ids."""
    labels_str = file_to_str(labels_uri)
    labels = json.loads(labels_str)
    filtered_features = []

    for feature in labels['features']:
        feature = copy.deepcopy(feature)
        properties = feature.get('properties')
        if properties:
            class_name = properties.get('class_name') or properties('label')
            if class_name in class_names:
                del properties['class_id']
                filtered_features.append(feature)

    new_labels = {'features': filtered_features}
    str_to_file(json.dumps(new_labels), output_uri)


if __name__ == '__main__':
    filter_geojson()
