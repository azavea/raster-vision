from os.path import join, dirname, basename
import os
import tempfile

from google.protobuf import text_format
from object_detection.protos.pipeline_pb2 import TrainEvalPipelineConfig
from keras_classification.protos.pipeline_pb2 import PipelineConfig

from rastervision.protos.model_config_pb2 import ModelConfig
from rastervision.protos.scene_pb2 import Scene
from rastervision.protos.raster_source_pb2 import RasterSource
from rastervision.protos.label_store_pb2 import LabelStore
from rastervision.protos.compute_stats_pb2 import (ComputeStatsConfig)
from rastervision.protos.make_chips_pb2 import (MakeChipsConfig)
from rastervision.protos.train_pb2 import TrainConfig
from rastervision.protos.predict_pb2 import PredictConfig
from rastervision.protos.eval_pb2 import EvalConfig
from rastervision.utils.files import (
    file_to_str, str_to_file, save_json_config, load_json_config,
    download_if_needed, upload_if_needed, file_exists, RV_TEMP_DIR)

OD = ModelConfig.Task.Value('OBJECT_DETECTION')
CL = ModelConfig.Task.Value('CLASSIFICATION')
TFOD = \
    ModelConfig.Backend.Value('TF_OBJECT_DETECTION_API')
KERAS = ModelConfig.Backend.Value('KERAS_CLASSIFICATION')

RESNET50 = \
    ModelConfig.ModelType.Value('RESNET50')
MOBILENET = ModelConfig.ModelType.Value('MOBILENET')

default_backends = {OD: TFOD, CL: KERAS}

default_model_types = {TFOD: MOBILENET, KERAS: RESNET50}

# yapf: disable
valid_task_backend_model_types = set([
    (OD, TFOD, MOBILENET),
    (CL, KERAS, RESNET50)])
# yapf: enable

# Note: There is no pretrained model specified for KERAS since it has its own
# mechanism for downloading and caching pretrained model files. In the future,
# we probably want to stop using that mechanism since it can't cache files on
# S3.
pretrained_model_uris = {
    TFOD: {
        MOBILENET:
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz'  # noqa
    },
    KERAS: {
        RESNET50: None
    }
}

BACKGROUND = 'background'

COMPUTE_STATS = 'compute-stats'
MAKE_CHIPS = 'make-chips'
TRAIN = 'train'
PREDICT = 'predict'
EVAL = 'eval'
ALL_COMMANDS = [COMPUTE_STATS, MAKE_CHIPS, TRAIN, PREDICT, EVAL]

backend_config_templates_path = join(
    dirname(__file__), 'backend-config-templates')


def make_stats_uri(compute_stats_uri):
    return join(compute_stats_uri, 'stats.json')


def make_model_uri(train_uri):
    return join(train_uri, 'model')


def make_prediction_package_uri(predict_uri):
    return join(predict_uri, 'prediction-package.zip')


def make_metrics_uri(eval_uri):
    return join(eval_uri, 'metrics.json')


def make_debug_uri(uri):
    return join(uri, 'debug')


def make_command_config_uri(command_uri, command):
    return join(command_uri, command + '-config.json')


def make_backend_config_uri(base_uri):
    return join(base_uri, 'backend_config.txt')


def make_experiment_config_uri(base_uri):
    return join(base_uri, 'experiment.json')


def make_class_items(names, colors=None):
    class_items = []
    for id, name in enumerate(names, 1):
        class_item = ModelConfig.ClassItem(id=id, name=name)
        if colors:
            class_item.color = colors[id - 1]
        class_items.append(class_item)
    return class_items


def get_default_backend(task):
    return default_backends[task]


def get_default_model_type(backend):
    return default_model_types[backend]


def is_valid_task_backend_model_type(task, backend, model_type):
    return (task, backend, model_type) in valid_task_backend_model_types


def make_model_config(class_names,
                      task,
                      chip_size,
                      backend=None,
                      model_type=None,
                      colors=None):
    model_config = ModelConfig()
    model_config.task = task
    model_config.chip_size = chip_size

    class_items = make_class_items(class_names, colors=colors)
    model_config.class_items.MergeFrom(class_items)

    model_config.backend = backend if backend else get_default_backend(task)
    model_config.model_type = (model_type if model_type else
                               get_default_model_type(model_config.backend))

    if not is_valid_task_backend_model_type(task, model_config.backend,
                                            model_config.model_type):
        raise ValueError(
            'Not a valid combination of task, backend, and model type.')

    return model_config


def make_geotiff_raster_source(raster_uris, stats_uri, channel_order=[0, 1,
                                                                      2]):
    raster_source = RasterSource()
    raster_source.geotiff_files.uris.extend(raster_uris)
    raster_source.raster_transformer.stats_uri = stats_uri
    raster_source.raster_transformer.channel_order.extend(channel_order)
    return raster_source


class ClassificationGeoJSONOptions():
    def __init__(self,
                 cell_size=300,
                 ioa_thresh=0.5,
                 use_intersection_over_cell=False,
                 background_class_id=None,
                 pick_min_class_id=True,
                 infer_cells=True):
        self.cell_size = cell_size
        self.ioa_thresh = ioa_thresh
        self.use_intersection_over_cell = use_intersection_over_cell
        self.background_class_id = background_class_id
        self.pick_min_class_id = pick_min_class_id
        self.infer_cells = infer_cells


class ObjectDetectionGeoJSONOptions():
    pass


def make_geojson_label_store(uri, task_options):
    label_store = LabelStore()

    if type(task_options) is int:
        if task_options == OD:
            task_options = ObjectDetectionGeoJSONOptions()
        elif task_options == CL:
            task_options = ClassificationGeoJSONOptions()

    if task_options is not None:
        if type(task_options) is ObjectDetectionGeoJSONOptions:
            label_store.object_detection_geojson_file.uri = uri
        elif type(task_options) is ClassificationGeoJSONOptions:
            label_store.classification_geojson_file.uri = uri
            options = label_store.classification_geojson_file.options
            options.ioa_thresh = task_options.ioa_thresh
            options.use_intersection_over_cell = \
                task_options.use_intersection_over_cell
            options.pick_min_class_id = task_options.pick_min_class_id
            if task_options.background_class_id is not None:
                options.background_class_id = task_options.background_class_id
            options.cell_size = task_options.cell_size
            options.infer_cells = task_options.infer_cells
        else:
            raise ValueError('Unknown type of task_options: ' +
                             str(type(task_options)))

    return label_store


def make_geotiff_geojson_scene(id,
                               raster_uris,
                               stats_uri,
                               task_options,
                               ground_truth_labels_uri=None,
                               prediction_base_uri=None,
                               channel_order=[0, 1, 2]):
    scene = Scene()
    scene.id = id
    scene.raster_source.MergeFrom(
        make_geotiff_raster_source(
            raster_uris, stats_uri, channel_order=channel_order))

    if ground_truth_labels_uri:
        scene.ground_truth_label_store.MergeFrom(
            make_geojson_label_store(ground_truth_labels_uri, task_options))

    if prediction_base_uri:
        predictions_uri = join(prediction_base_uri, str(id) + '.json')
        scene.prediction_label_store.MergeFrom(
            make_geojson_label_store(predictions_uri, task_options))

    return scene


def get_cache_uri(model_uri):
    """Get the URI of cached version of model file.

    Rather than download pretrained model files off the web every time we use
    them, this attempts to download it from a cache
    (specified by the RV_MODEL_CACHE env var). If it's not in the cache,
    this attempts to update the cache.

    Args:
        model_uri: URI to a model file

    Returns:
        URI of cached copy of model file
    """
    if not model_uri:
        return None
    default_cache_dir = join(RV_TEMP_DIR, 'model-cache')
    cache_uri = os.environ.get('RV_MODEL_CACHE', default_cache_dir)
    fname = basename(model_uri)
    cache_model_uri = join(cache_uri, fname)
    if not file_exists(cache_model_uri):
        print('Updating cache...')
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = download_if_needed(model_uri, temp_dir)
            upload_if_needed(model_path, cache_model_uri)
    print('Using cached model URI: ' + cache_model_uri)
    return cache_model_uri


def get_pretrained_model_uri(backend, model_type=None):
    if model_type is None:
        model_type = get_default_model_type(backend)
    model_uri = pretrained_model_uris[backend][model_type]
    return get_cache_uri(model_uri)


def save_mobilenet_config(backend_config_uri, chip_size, num_classes,
                          batch_size, num_steps):
    sample_path = join(backend_config_templates_path,
                       'tf-object-detection-api', 'mobilenet.config')
    config = text_format.Parse(
        file_to_str(sample_path), TrainEvalPipelineConfig())

    config.model.ssd.image_resizer.fixed_shape_resizer.height = chip_size
    config.model.ssd.image_resizer.fixed_shape_resizer.width = chip_size
    config.model.ssd.num_classes = num_classes
    config.train_config.batch_size = batch_size
    config.train_config.num_steps = num_steps
    str_to_file(text_format.MessageToString(config), backend_config_uri)


def save_resnet50_config(backend_config_uri, chip_size, class_names,
                         batch_size, num_epochs):
    sample_path = join(backend_config_templates_path, 'keras-classification',
                       'resnet50.json')
    config = load_json_config(sample_path, PipelineConfig())

    config.model.input_size = chip_size
    config.model.nb_classes = len(class_names)
    config.trainer.options.nb_epochs = num_epochs
    config.trainer.options.batch_size = batch_size
    config.trainer.options.input_size = chip_size
    del config.trainer.options.class_names[:]
    config.trainer.options.class_names.extend(class_names)
    save_json_config(config, backend_config_uri)


def save_backend_config(backend_config_uri,
                        backend,
                        chip_size,
                        class_names,
                        batch_size,
                        num_iters,
                        model_type=None):
    """Save a backend config file.

    Depending on backend and model_type, picks a template of a backend config
    file, modifies it according to the args, and saves it. This only supports
    the most commonly varied hyperparameters. Tweak other hyperparameters
    requires making a copy of the template and modifying it manually.
    """
    if model_type is None:
        model_type = get_default_model_type(backend)

    if backend == TFOD and model_type == MOBILENET:
        save_mobilenet_config(
            backend_config_uri,
            chip_size,
            len(class_names),
            batch_size=batch_size,
            num_steps=num_iters)
    elif backend == KERAS and model_type == RESNET50:
        save_resnet50_config(
            backend_config_uri,
            chip_size,
            class_names,
            batch_size=batch_size,
            num_epochs=num_iters)


def make_compute_stats(raster_sources, output_uri):
    config = ComputeStatsConfig()
    config.raster_sources.MergeFrom(raster_sources)
    config.stats_uri = make_stats_uri(output_uri)
    return config


class ObjectDetectionMakeChipsOptions():
    def __init__(self,
                 neg_ratio=1.0,
                 ioa_thresh=0.8,
                 window_method='chip',
                 label_buffer=0.0):
        self.neg_ratio = neg_ratio
        self.ioa_thresh = ioa_thresh
        self.window_method = window_method
        self.label_buffer = label_buffer


def make_make_chips(train_scenes,
                    validation_scenes,
                    model_config,
                    output_uri,
                    task_options,
                    chip_size=300,
                    debug=True):
    config = MakeChipsConfig()
    config.train_scenes.MergeFrom(train_scenes)
    config.validation_scenes.MergeFrom(validation_scenes)
    config.model_config.MergeFrom(model_config)
    config.options.chip_size = chip_size
    config.options.debug = debug
    config.options.output_uri = output_uri

    if type(task_options) is int:
        if task_options == OD:
            task_options = ObjectDetectionMakeChipsOptions()
        elif task_options == CL:
            task_options = None
        else:
            raise ValueError('Unknown task: ' + task_options)

    if task_options is not None:
        if type(task_options) is ObjectDetectionMakeChipsOptions:
            config.options.object_detection_options.neg_ratio = \
                task_options.neg_ratio
            config.options.object_detection_options.ioa_thresh = \
                task_options.ioa_thresh
            config.options.object_detection_options.window_method = \
                task_options.window_method
            config.options.object_detection_options.label_buffer = \
                task_options.label_buffer
        else:
            raise ValueError('Unknown type of task_options: ' +
                             str(type(task_options)))

    return config


def make_train(model_config,
               backend_config_uri,
               make_chips_uri,
               output_uri,
               pretrained_model_uri=None,
               sync_interval=600):
    config = TrainConfig()
    config.model_config.MergeFrom(model_config)
    config.options.backend_config_uri = backend_config_uri
    config.options.training_data_uri = make_chips_uri
    config.options.output_uri = output_uri
    if pretrained_model_uri is not None:
        config.options.pretrained_model_uri = pretrained_model_uri
    config.options.sync_interval = sync_interval
    return config


class ObjectDetectionPredictOptions():
    def __init__(self, merge_thresh=0.5, score_thresh=0.5):
        self.merge_thresh = merge_thresh
        self.score_thresh = score_thresh


def make_predict(model_config,
                 scenes,
                 chip_size,
                 train_uri,
                 output_uri,
                 task_options,
                 debug=True):
    config = PredictConfig()
    config.model_config.MergeFrom(model_config)
    config.scenes.MergeFrom(scenes)
    config.options.debug = debug
    config.options.debug_uri = make_debug_uri(output_uri)
    config.options.chip_size = chip_size
    config.options.model_uri = make_model_uri(train_uri)
    config.options.prediction_package_uri = \
        make_prediction_package_uri(output_uri)

    if type(task_options) is int:
        if task_options == OD:
            task_options = ObjectDetectionPredictOptions()
        elif task_options == CL:
            task_options = None
        else:
            raise ValueError('Unknown task: ' + task_options)

    if task_options is not None:
        if type(task_options) is ObjectDetectionPredictOptions:
            config.options.object_detection_options.merge_thresh = \
                task_options.merge_thresh
            config.options.object_detection_options.score_thresh = \
                task_options.score_thresh
        else:
            raise ValueError('Unknown type of task_options: ' +
                             str(type(task_options)))
    return config


def make_eval(model_config, scenes, output_uri, debug=True):
    config = EvalConfig()
    config.model_config.MergeFrom(model_config)
    config.scenes.MergeFrom(scenes)
    config.options.debug = debug
    config.options.output_uri = make_metrics_uri(output_uri)
    return config
