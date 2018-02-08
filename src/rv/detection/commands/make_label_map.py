import json


def get_labels(label_paths):
    labels = set()
    for label_path in label_paths:
        with open(label_path, 'r') as label_file:
            geojson = json.load(label_file)
            features = geojson['features']

            for feature in features:
                if 'properties' in feature:
                    if 'label' in feature['properties']:
                        labels.add(feature['properties']['label'])
                    else:
                        labels.add('Unknown')

    labels = sorted(list(labels))
    return labels


def write_label_map(path, labels):
    with open(path, 'w') as label_map_file:
        for class_id, label in enumerate(labels, start=1):
            item_str = '''
item {{
  id: {}
  name: \'{}\'
}}
'''.format(class_id, label)
            label_map_file.write(item_str)


def make_label_map(label_paths, label_map_path):
    """Generate a label map based on the labels in a set of label files.

    Args:
        label_paths: List of paths to GeoJSON files containing labels
        label_map_path: Path to Protobuf file with map of labels to ids
    """
    labels = get_labels(label_paths)
    write_label_map(label_map_path, labels)
