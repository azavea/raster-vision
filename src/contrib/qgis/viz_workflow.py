"""QGIS 3.0 script to visualize predictions and eval from a workflow config.

Given a workflow config and the output of running the workflow, this
script will add styled raster and vector layers to compare the predictions to
the ground truth for each of the test scenes. It also prints the eval to the
console. To run this script, it should be copied into the QGIS Python console,
and the rv_root and workflow_path variables at the bottom should be set.
"""
import os
import json


def make_vector_renderer(layer, class_field, class_items):
    category_map = {}

    for class_item in class_items:
        name = class_item['name']
        color = class_item.get('color', 'Red')
        category_map[name] = (color, name)

    categories = []
    for name, (color, label) in category_map.items():
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol_layer = QgsSimpleLineSymbolLayer()
        symbol_layer.setWidth(0.5)
        symbol.changeSymbolLayer(0, symbol_layer)
        symbol.setColor(QColor(color))

        category = QgsRendererCategory(label, symbol, label)
        categories.append(category)

    renderer = QgsCategorizedSymbolRenderer(class_field, categories)
    return renderer


def get_class_field(labels_uri):
    with open(labels_uri, 'r') as labels_file:
        labels = json.load(labels_file)
    feature = labels['features'][0]
    properties = feature.get('properties', {})
    if 'class_name' in properties:
        return 'class_name'
    return 'label'


def clear_layers():
    layer_ids = QgsProject.instance().mapLayers().keys()
    for layer_id in layer_ids:
        QgsProject.instance().removeMapLayer(layer_id)


def dump_eval(eval_uri):
    with open(eval_uri, 'r') as eval_file:
        eval = json.load(eval_file)
        print(json.dumps(eval, indent=2))


def set_channel_order(raster_layer, channel_order):
    # Based on https://gis.stackexchange.com/questions/171773/displaying-rgb-raster-on-qgis-with-no-enhancement-with-python-console?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    renderer = raster_layer.renderer()
    renderer.setRedBand(channel_order[0])
    renderer.setGreenBand(channel_order[1])
    renderer.setBlueBand(channel_order[2])
    bands = renderer.usesBands()

    ces = []
    for band in bands:
        min = raster_layer.dataProvider().bandStatistics(band, QgsRasterBandStats.All).minimumValue
        max = raster_layer.dataProvider().bandStatistics(band, QgsRasterBandStats.All).maximumValue
        type = raster_layer.renderer().dataType(band)
        ce = QgsContrastEnhancement(type)
        ce.setContrastEnhancementAlgorithm(
            QgsContrastEnhancement.StretchToMinimumMaximum, True)
        ce.setMinimumValue(min)
        ce.setMaximumValue(max)
        ces.append(ce)
    renderer.setRedContrastEnhancement(ces[0])
    renderer.setGreenContrastEnhancement(ces[1])
    renderer.setBlueContrastEnhancement(ces[2])


def viz_scenes(workflow):
    channel_order = workflow['raster_transformer']['channel_order']
    for scene in workflow['test_scenes']:
        id = scene['id']
        is_classification = workflow['machine_learning']['task'] == 'CLASSIFICATION'
        key = 'classification_geojson_file' \
            if is_classification else 'object_detection_geojson_file'
        class_items = workflow['machine_learning']['class_items']

        raster_uris = scene['raster_source']['geotiff_files']['uris']
        raster_uris = [uri.format(rv_root=rv_root) for uri in raster_uris]
        for raster_uri in raster_uris:
            raster_layer = iface.addRasterLayer(raster_uri, id)
            # TODO get this working.
            # set_channel_order(raster_layer, channel_order)

        gt_labels_uri = (
            scene['ground_truth_label_store'][key]
            ['uri'].format(rv_root=rv_root))
        gt_layer = iface.addVectorLayer(
            gt_labels_uri, 'ground-truth-' + id, 'ogr')
        class_field = get_class_field(gt_labels_uri)
        renderer = make_vector_renderer(gt_layer, class_field, class_items)
        gt_layer.setRenderer(renderer)

        # TODO use different symbol for preds
        prediction_labels_uri = os.path.join(
                rv_root, 'rv-output', 'raw-datasets', workflow['raw_dataset_key'],
                'datasets', workflow['dataset_key'], 'models', workflow['model_key'],
                'predictions', workflow['prediction_key'], 'output', id + '.json')
        pred_layer = iface.addVectorLayer(
            prediction_labels_uri, 'predictions-' + id, 'ogr')
        class_field = get_class_field(prediction_labels_uri)
        renderer = make_vector_renderer(pred_layer, class_field, class_items)
        pred_layer.setRenderer(renderer)


def viz_workflow(workflow_path):
    clear_layers()

    with open(workflow_path, 'r') as workflow_file:
        workflow = json.load(workflow_file)

    eval_uri = os.path.join(
        rv_root, 'rv-output', 'raw-datasets', workflow['raw_dataset_key'],
        'datasets', workflow['dataset_key'], 'models', workflow['model_key'],
        'predictions', workflow['prediction_key'], 'evals',
        workflow['eval_key'], 'output', 'eval.json')
    dump_eval(eval_uri)

    viz_scenes(workflow)


# Fill in path to rv_root directory (one level above rv-output)
rv_root = ''
# Fill in path to workflow after downloading rv-output to rv-root from remote
# storage.
workflow_path = ''
viz_workflow(workflow_path)
