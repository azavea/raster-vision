import os
from copy import deepcopy
from google.protobuf import (text_format, json_format)

import rastervision as rv
from rastervision.backend import (BackendConfig, BackendConfigBuilder,
                                  TFObjectDetection)
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg
from rastervision.protos.tf_object_detection.pipeline_pb2 import TrainEvalPipelineConfig
from rastervision.utils.files import file_to_str
from rastervision.utils.misc import set_nested_keys

# Default location to Tensorflow Object Detection's scripts.
DEFAULT_SCRIPT_TRAIN = '/opt/tf-models/object_detection/model_main.py'
DEFAULT_SCRIPT_EXPORT = '/opt/tf-models/object_detection/export_inference_graph.py'
CHIP_OUTPUT_FILES = ['label-map.pbtxt', 'train.record', 'validation.record']
DEBUG_CHIP_OUTPUT_FILES = [
    'train-debug-chips.zip', 'validation-debug-chips.zip'
]


class TFObjectDetectionConfig(BackendConfig):
    class TrainOptions:
        def __init__(self,
                     sync_interval=600,
                     do_monitoring=True,
                     replace_model=False):
            self.sync_interval = sync_interval
            self.do_monitoring = do_monitoring
            self.replace_model = replace_model

    class ScriptLocations:
        def __init__(self,
                     model_main_uri=DEFAULT_SCRIPT_TRAIN,
                     export_uri=DEFAULT_SCRIPT_EXPORT):
            self.model_main_uri = model_main_uri
            self.export_uri = export_uri

    def __init__(self,
                 tfod_config,
                 pretrained_model_uri=None,
                 train_options=None,
                 script_locations=None,
                 debug=False,
                 training_data_uri=None,
                 training_output_uri=None,
                 model_uri=None,
                 fine_tune_checkpoint_name=None):
        if train_options is None:
            train_options = TFObjectDetectionConfig.TrainOptions()
        if script_locations is None:
            script_locations = TFObjectDetectionConfig.ScriptLocations()

        super().__init__(rv.TF_OBJECT_DETECTION, pretrained_model_uri)
        self.tfod_config = tfod_config
        self.pretrained_model_uri = pretrained_model_uri
        self.train_options = train_options
        self.script_locations = script_locations
        self.debug = debug

        # Internally set from command preprocessing
        self.training_data_uri = training_data_uri
        self.training_output_uri = training_output_uri
        self.model_uri = model_uri
        self.fine_tune_checkpoint_name = fine_tune_checkpoint_name

    def create_backend(self, task_config):
        return TFObjectDetection(self, task_config)

    def get_num_steps(self):
        return int(self.tfod_config['trainConfig']['numSteps'])

    def get_batch_size(self, batch_size):
        return int(self.tfod_config.train_config['trainConfig']['batchSize'])

    def to_proto(self):
        d = {
            'sync_interval': self.train_options.sync_interval,
            'do_monitoring': self.train_options.do_monitoring,
            'replace_model': self.train_options.replace_model,
            'model_main_py': self.script_locations.model_main_uri,
            'export_py': self.script_locations.export_uri,
            'training_data_uri': self.training_data_uri,
            'training_output_uri': self.training_output_uri,
            'model_uri': self.model_uri,
            'debug': self.debug,
            'fine_tune_checkpoint_name': self.fine_tune_checkpoint_name,
            'tfod_config': self.tfod_config
        }

        conf = json_format.ParseDict(
            d, BackendConfigMsg.TFObjectDetectionConfig())

        msg = BackendConfigMsg(
            backend_type=rv.TF_OBJECT_DETECTION,
            tf_object_detection_config=conf)

        if self.pretrained_model_uri:
            msg.MergeFrom(
                BackendConfigMsg(
                    pretrained_model_uri=self.pretrained_model_uri))

        return msg

    def save_bundle_files(self, bundle_dir):
        if not self.model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_path, base_name = self.bundle_file(self.model_uri, bundle_dir)
        new_config = self.to_builder() \
                         .with_model_uri(base_name) \
                         .build()
        return (new_config, [local_path])

    def load_bundle_files(self, bundle_dir):
        if not self.model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_model_uri = os.path.join(bundle_dir, self.model_uri)
        return self.to_builder() \
                   .with_model_uri(local_model_uri) \
                   .build()

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        conf, io_def = super().update_for_command(command_type,
                                                  experiment_config, context)
        if command_type == rv.CHIP:
            if not conf.training_data_uri:
                conf.training_data_uri = experiment_config.chip_uri

            outputs = list(
                map(lambda x: os.path.join(conf.training_data_uri, x),
                    CHIP_OUTPUT_FILES))

            if self.debug:
                outputs.extend(
                    list(
                        map(lambda x: os.path.join(conf.training_data_uri, x),
                            DEBUG_CHIP_OUTPUT_FILES)))

            io_def.add_outputs(outputs)
        if command_type == rv.TRAIN:
            if not conf.training_data_uri:
                io_def.add_missing('Missing training_data_uri.')
            else:
                inputs = list(
                    map(lambda x: os.path.join(conf.training_data_uri, x),
                        CHIP_OUTPUT_FILES))
                io_def.add_inputs(inputs)

            if not conf.training_output_uri:
                conf.training_output_uri = experiment_config.train_uri

            if not conf.model_uri:
                conf.model_uri = os.path.join(conf.training_output_uri,
                                              'model')
            io_def.add_output(conf.model_uri)

            # Set the fine tune checkpoint name to the experiment id
            if not conf.fine_tune_checkpoint_name:
                conf.fine_tune_checkpoint_name = experiment_config.id
            io_def.add_output(conf.fine_tune_checkpoint_name)
        if command_type in [rv.PREDICT, rv.BUNDLE]:
            if not conf.model_uri:
                io_def.add_missing('Missing model_uri.')
            else:
                io_def.add_input(conf.model_uri)

        return (conf, io_def)


class TFObjectDetectionConfigBuilder(BackendConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'tfod_config': prev.tfod_config,
                'pretrained_model_uri': prev.pretrained_model_uri,
                'train_options': prev.train_options,
                'script_locations': prev.script_locations,
                'debug': prev.debug,
                'training_data_uri': prev.training_data_uri,
                'training_output_uri': prev.training_output_uri,
                'model_uri': prev.model_uri,
                'fine_tune_checkpoint_name': prev.fine_tune_checkpoint_name
            }
        super().__init__(rv.TF_OBJECT_DETECTION, TFObjectDetectionConfig,
                         config, prev)
        self.config_mods = []
        self.require_task = prev is None

    def from_proto(self, msg):
        b = super().from_proto(msg)
        conf = msg.tf_object_detection_config
        # Since this is coming from a serialized message,
        # assume the task has already been set and do not
        # require it during validation.
        b.require_task = False
        b = b.with_train_options(
            sync_interval=conf.sync_interval,
            do_monitoring=conf.do_monitoring,
            replace_model=conf.replace_model)
        b = b.with_script_locations(
            model_main_uri=conf.model_main_py, export_uri=conf.export_py)
        b = b.with_training_data_uri(conf.training_data_uri)
        b = b.with_training_output_uri(conf.training_output_uri)
        b = b.with_model_uri(conf.model_uri)
        b = b.with_fine_tune_checkpoint_name(conf.fine_tune_checkpoint_name)
        b = b.with_debug(conf.debug)

        return b.with_template(json_format.MessageToDict(conf.tfod_config))

    def validate(self):
        super().validate()
        if not self.config.get('tfod_config'):
            raise rv.ConfigError('You must specify a template for the backend '
                                 "configuration - use 'with_template'.")
        if self.require_task and not self.task:
            raise rv.ConfigError('You must specify the task this backend '
                                 "is for - use 'with_task'.")
        return True

    def build(self):
        """Build this configuration, setting any values into the
           TF object detection pipeline config as necessary.
        """
        self.validate()

        b = deepcopy(self)

        # Check if a pretrained model was assigned.
        pretrained_model = b.config.get('pretrained_model_uri')
        if pretrained_model:
            b = b.with_config({'fineTuneCheckpoint': pretrained_model})
        else:
            b = b.with_config(
                {
                    'fineTuneCheckpoint': ''
                }, ignore_missing_keys=True)

        for config_mod, ignore_missing_keys, set_missing_keys in b.config_mods:
            try:
                set_nested_keys(b.config['tfod_config'], config_mod,
                                ignore_missing_keys, set_missing_keys)
            except Exception as e:
                raise rv.ConfigError(
                    'Error setting configuration {}'.format(config_mod)) from e

        return TFObjectDetectionConfig(**b.config)

    def _applicable_tasks(self):
        return [rv.OBJECT_DETECTION]

    def _process_task(self):
        return self.with_config(
            {
                'numClasses': len(self.task.class_map.get_items()),
                'imageResizer': {
                    'fixedShapeResizer': {
                        'height': self.task.chip_size,
                        'width': self.task.chip_size
                    },
                    'keepAspectRatioResizer': {
                        'minDimension': self.task.chip_size,
                        'maxDimension': self.task.chip_size
                    }
                }
            },
            ignore_missing_keys=True)

    def _load_model_defaults(self, model_defaults):
        """Loads defaults. Expected keys are "pretrained_model_uri" and "pipeline_config_uri",
           neither of which is required.
        """
        expected_keys = ['pretrained_model_uri', 'pipeline_config_uri']
        unknown_keys = set(model_defaults.keys()) - set(expected_keys)
        if unknown_keys:
            raise rv.ConfigError('Unexpected keys in model defaults:'
                                 ' {}. Expected keys: {}'.format(
                                     unknown_keys, expected_keys))

        b = self
        if 'pretrained_model_uri' in model_defaults:
            b = b.with_pretrained_model(model_defaults['pretrained_model_uri'])
        if 'pipeline_config_uri' in model_defaults:
            b = b.with_template(model_defaults['pipeline_config_uri'])
        return b

    def with_template(self, template):
        """Use a template for TF Object Detection pipeline config.

        Args:
           template: A dict, string or uri as the base for the tensorflow object
                     detection API model training pipeline, for example those found
                     here:
                     https://github.com/tensorflow/models/tree/eef6bb5bd3b3cd5fcf54306bf29750b7f9f9a5ea/research/object_detection/samples/configs # noqa
        """

        template_json = None
        if type(template) is dict:
            template_json = template
        else:
            # Try parsing the string as a message, on fail assume it's a URI
            msg = None
            try:
                msg = text_format.Parse(template, TrainEvalPipelineConfig())
            except text_format.ParseError:
                msg = text_format.Parse(
                    file_to_str(template), TrainEvalPipelineConfig())
            template_json = json_format.MessageToDict(msg)

        b = deepcopy(self)
        b.config['tfod_config'] = template_json
        return b

    def with_batch_size(self, batch_size):
        """Sets the training batch size."""
        return self.with_config(
            {
                'trainConfig': {
                    'batchSize': batch_size
                }
            }, set_missing_keys=True)

    def with_num_steps(self, num_steps):
        """Sets the number of training steps."""
        return self.with_config(
            {
                'trainConfig': {
                    'numSteps': num_steps
                }
            }, set_missing_keys=True)

    def with_config(self,
                    config_mod,
                    ignore_missing_keys=False,
                    set_missing_keys=False):
        """Given a dict, modify the tensorflow pipeline configuration
           such that keys that are found recursively in the configuration
           are replaced with those values. TODO: better explination.
        """
        b = deepcopy(self)
        b.config_mods.append((config_mod, ignore_missing_keys,
                              set_missing_keys))
        return b

    def with_debug(self, debug):
        """Sets the debug flag for this backend.
        """
        b = deepcopy(self)
        b.config['debug'] = debug
        return b

    def with_training_data_uri(self, training_data_uri):
        """Whence comes the training data?

            Args:
                training_data_uri: The location of the training data.

        """
        b = deepcopy(self)
        b.config['training_data_uri'] = training_data_uri
        return b

    def with_training_output_uri(self, training_output_uri):
        """Whither goes the training output?

            Args:
                training_output_uri: The location where the training
                    output will be stored.

        """
        b = deepcopy(self)
        b.config['training_output_uri'] = training_output_uri
        return b

    def with_model_uri(self, model_uri):
        """Defines the name of the model file that will be created for
        this model after training.

        """
        b = deepcopy(self)
        b.config['model_uri'] = model_uri
        return b

    def with_fine_tune_checkpoint_name(self, fine_tune_checkpoint_name):
        """Defines the name of the fine tune checkpoint that will
        be created for this model after training."""
        b = deepcopy(self)
        b.config['fine_tune_checkpoint_name'] = fine_tune_checkpoint_name
        return b

    def with_train_options(self,
                           sync_interval=600,
                           do_monitoring=True,
                           replace_model=False):
        """Sets the train options for this backend.

           Args:
              sync_interval: How often to sync output of training
                             to the cloud (in seconds).

              do_monitoring: Run process to monitor training (eg. Tensorboard)

              replace_model: Replace the model checkpoint if exists.
                             If false, this will continue training from
                             checkpoing if exists, if the backend allows for this.
        """
        b = deepcopy(self)
        b.config['train_options'] = TFObjectDetectionConfig.TrainOptions(
            sync_interval, do_monitoring, replace_model)
        return b

    def with_script_locations(self,
                              model_main_uri=DEFAULT_SCRIPT_TRAIN,
                              export_uri=DEFAULT_SCRIPT_EXPORT):
        sl = TFObjectDetectionConfig.ScriptLocations(model_main_uri,
                                                     export_uri)
        b = deepcopy(self)
        b.config['script_locations'] = sl
        return b
