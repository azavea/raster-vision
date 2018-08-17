from os.path import join
import copy

from rastervision.config.utils import (
    make_geotiff_geojson_scene, make_compute_stats, make_make_chips,
    make_train, make_predict, make_eval, make_stats_uri,
    make_command_config_uri, ClassificationGeoJSONOptions, CL, COMPUTE_STATS,
    MAKE_CHIPS, TRAIN, PREDICT, EVAL, make_model_config,
    get_pretrained_model_uri, save_backend_config, make_backend_config_uri,
    make_experiment_config_uri)
from rastervision.utils.files import save_json_config
from rastervision.protos.experiment_pb2 import ExperimentConfig

BACKGROUND = 'background'


class ExperimentPaths():
    """The paths to the output directories for each command."""

    def __init__(self,
                 experiment_uri,
                 compute_stats_uri=None,
                 make_chips_uri=None,
                 train_uri=None,
                 predict_uri=None,
                 eval_uri=None):
        self.experiment_uri = experiment_uri
        self.compute_stats_uri = (compute_stats_uri
                                  if compute_stats_uri else join(
                                      experiment_uri, COMPUTE_STATS))
        self.make_chips_uri = (make_chips_uri if make_chips_uri else join(
            experiment_uri, MAKE_CHIPS))
        self.train_uri = (train_uri
                          if train_uri else join(experiment_uri, TRAIN))
        self.predict_uri = (predict_uri
                            if predict_uri else join(experiment_uri, PREDICT))
        self.eval_uri = (eval_uri if eval_uri else join(experiment_uri, EVAL))


class ExperimentHelper():
    """High-level API for making protobufs related to an experiment.

    This should be used to make various protobufs that are then consumed by
    the Experiment constructor.
    """

    def __init__(self, paths, chip_size=300):
        self.paths = paths
        self.chip_size = chip_size

    def make_scene(self,
                   id,
                   model_config,
                   raster_uris,
                   task_options,
                   ground_truth_labels_uri=None,
                   channel_order=[0, 1, 2]):
        if task_options == CL:
            background_class_id = None
            for item in model_config.class_items:
                if item.name == BACKGROUND:
                    background_class_id = item.id
            task_options = ClassificationGeoJSONOptions(
                background_class_id=background_class_id)

        if type(task_options) is ClassificationGeoJSONOptions:
            # Force cell_size to be consistent with rest of experiment.
            task_options = copy.deepcopy(task_options)
            task_options.cell_size = self.chip_size

        return make_geotiff_geojson_scene(
            id,
            raster_uris,
            make_stats_uri(self.paths.compute_stats_uri),
            task_options,
            ground_truth_labels_uri=ground_truth_labels_uri,
            prediction_base_uri=self.paths.predict_uri,
            channel_order=channel_order)

    def make_model_config(self,
                          class_names,
                          task,
                          backend=None,
                          model_type=None,
                          colors=None,
                          add_background_for_cl=True):
        if task == CL and add_background_for_cl:
            class_names = class_names + [BACKGROUND]
        return make_model_config(
            class_names,
            task,
            self.chip_size,
            backend=backend,
            model_type=model_type,
            colors=colors)

    def get_backend_config_uri(self, model_config, batch_size, num_iters):
        class_names = [
            class_item.name for class_item in model_config.class_items
        ]
        backend_config_uri = make_backend_config_uri(self.paths.experiment_uri)
        save_backend_config(
            backend_config_uri,
            model_config.backend,
            self.chip_size,
            class_names,
            batch_size,
            num_iters,
            model_type=model_config.model_type)
        return backend_config_uri

    def get_pretrained_model_uri(self, model_config):
        return get_pretrained_model_uri(
            model_config.backend, model_type=model_config.model_type)


class Experiment(object):
    """High-level API for making Experiment protobufs.

    An experiment encapsulates a sequence of commands that are configured to
    work together. The sequence is `compute_stats`, `make_chips`, `train`,
    `predict`, and `eval`.
    """

    def __init__(self,
                 paths,
                 model_config,
                 train_scenes,
                 validation_scenes,
                 backend_config_uri,
                 pretrained_model_uri,
                 sync_interval=600,
                 test_scenes=None,
                 debug=True,
                 task_make_chips_options=None,
                 task_predict_options=None):
        self.paths = paths
        self.model_config = model_config
        self.train_scenes = train_scenes
        self.validation_scenes = validation_scenes

        self.backend_config_uri = backend_config_uri
        self.pretrained_model_uri = pretrained_model_uri
        self.sync_interval = sync_interval
        self.test_scenes = [] if test_scenes is None else test_scenes
        self.debug = debug
        self.task_make_chips_options = (model_config.task
                                        if task_make_chips_options is None else
                                        task_make_chips_options)
        self.task_predict_options = (model_config.task
                                     if task_predict_options is None else
                                     task_predict_options)

    def make_compute_stats(self):
        scenes = self.train_scenes + self.validation_scenes + self.test_scenes
        # Strip stats_uri from raster_sources since the stats won't have
        # been computed yet since this is the compute_stats command.
        raster_sources = []
        for scene in scenes:
            raster_source = copy.deepcopy(scene.raster_source)
            raster_source.raster_transformer.stats_uri = ''
            raster_sources.append(raster_source)
        return make_compute_stats(raster_sources, self.paths.compute_stats_uri)

    def make_make_chips(self):
        return make_make_chips(
            self.train_scenes,
            self.validation_scenes,
            self.model_config,
            self.paths.make_chips_uri,
            self.task_make_chips_options,
            chip_size=self.model_config.chip_size,
            debug=self.debug)

    def make_train(self):
        return make_train(self.model_config, self.backend_config_uri,
                          self.paths.make_chips_uri, self.paths.train_uri,
                          self.pretrained_model_uri, self.sync_interval)

    def make_predict(self):
        return make_predict(
            self.model_config,
            self.validation_scenes + self.test_scenes,
            self.model_config.chip_size,
            self.paths.train_uri,
            self.paths.predict_uri,
            self.task_predict_options,
            debug=self.debug)

    def make_eval(self):
        return make_eval(
            self.model_config,
            self.validation_scenes,
            self.paths.eval_uri,
            debug=self.debug)

    def save(self):
        config = ExperimentConfig()

        config.model_config.MergeFrom(self.model_config)
        config.train_scenes.MergeFrom(self.train_scenes)
        config.validation_scenes.MergeFrom(self.validation_scenes)
        config.test_scenes.MergeFrom(self.test_scenes)

        config.compute_stats_uri = self.paths.compute_stats_uri
        config.make_chips_uri = self.paths.make_chips_uri
        config.train_uri = self.paths.train_uri
        config.predict_uri = self.paths.predict_uri
        config.eval_uri = self.paths.eval_uri

        config.compute_stats_config_uri = make_command_config_uri(
            self.paths.compute_stats_uri, COMPUTE_STATS)
        config.make_chips_config_uri = make_command_config_uri(
            self.paths.make_chips_uri, MAKE_CHIPS)
        config.train_config_uri = make_command_config_uri(
            self.paths.train_uri, TRAIN)
        config.predict_config_uri = make_command_config_uri(
            self.paths.predict_uri, PREDICT)
        config.eval_config_uri = make_command_config_uri(
            self.paths.eval_uri, EVAL)

        config.compute_stats.MergeFrom(self.make_compute_stats())
        config.make_chips.MergeFrom(self.make_make_chips())
        config.train.MergeFrom(self.make_train())
        config.predict.MergeFrom(self.make_predict())
        config.eval.MergeFrom(self.make_eval())

        experiment_config_uri = make_experiment_config_uri(
            self.paths.experiment_uri)
        save_json_config(config, experiment_config_uri)
        print('Wrote experiment config to: ' + experiment_config_uri)
