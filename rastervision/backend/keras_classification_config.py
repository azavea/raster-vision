import os
from copy import deepcopy
from google.protobuf import (json_format)

import rastervision as rv
from rastervision.backend import (BackendConfig, BackendConfigBuilder)
from rastervision.utils.misc import set_nested_keys
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg
from rastervision.utils.files import file_to_str
from rastervision.protos.keras_classification.pipeline_pb2 import PipelineConfig

# Default location to Tensorflow Object Detection's scripts.
CHIP_OUTPUT_FILES = ['training.zip', 'validation.zip']


class KerasClassificationConfig(BackendConfig):
    class TrainOptions:
        def __init__(self,
                     sync_interval=600,
                     do_monitoring=True,
                     replace_model=False):
            self.sync_interval = sync_interval
            self.do_monitoring = do_monitoring
            self.replace_model = replace_model

    def __init__(self,
                 kc_config,
                 pretrained_model_uri=None,
                 train_options=None,
                 debug=False,
                 training_data_uri=None,
                 training_output_uri=None,
                 model_uri=None):
        if train_options is None:
            train_options = KerasClassificationConfig.TrainOptions()

        super().__init__(rv.KERAS_CLASSIFICATION, pretrained_model_uri)
        self.kc_config = kc_config
        self.pretrained_model_uri = pretrained_model_uri
        self.train_options = train_options
        self.debug = debug

        # Internally set from command preprocessing
        self.training_data_uri = training_data_uri
        self.training_output_uri = training_output_uri
        self.model_uri = model_uri

    def create_backend(self, task_config):
        from rastervision.backend.keras_classification import KerasClassification
        return KerasClassification(self, task_config)

    def to_proto(self):
        d = {
            'sync_interval': self.train_options.sync_interval,
            'do_monitoring': self.train_options.do_monitoring,
            'replace_model': self.train_options.replace_model,
            'training_data_uri': self.training_data_uri,
            'training_output_uri': self.training_output_uri,
            'model_uri': self.model_uri,
            'debug': self.debug,
            'kc_config': self.kc_config
        }

        conf = json_format.ParseDict(
            d, BackendConfigMsg.KerasClassificationConfig())

        msg = BackendConfigMsg(
            backend_type=rv.KERAS_CLASSIFICATION,
            keras_classification_config=conf)

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

        if command_type in [rv.PREDICT, rv.BUNDLE]:
            if not conf.model_uri:
                io_def.add_missing('Missing model_uri.')
            else:
                io_def.add_input(conf.model_uri)

        return (conf, io_def)


class KerasClassificationConfigBuilder(BackendConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'kc_config': prev.kc_config,
                'pretrained_model_uri': prev.pretrained_model_uri,
                'train_options': prev.train_options,
                'debug': prev.debug,
                'training_data_uri': prev.training_data_uri,
                'training_output_uri': prev.training_output_uri,
                'model_uri': prev.model_uri
            }
        super().__init__(rv.KERAS_CLASSIFICATION, KerasClassificationConfig,
                         config, prev)
        self.config_mods = []
        self.require_task = prev is None

    def from_proto(self, msg):
        b = super().from_proto(msg)
        conf = msg.keras_classification_config
        # Since this is coming from a serialized message,
        # assume the task has already been set and do not
        # require it during validation.
        b.require_task = False
        if self.config.get('pretrained_model_uri'):
            b = b.with_pretrained_model_uri(self.config.pretrained_model_uri)
        b = b.with_train_options(
            sync_interval=conf.sync_interval,
            do_monitoring=conf.do_monitoring,
            replace_model=conf.replace_model,
        )
        b = b.with_debug(conf.debug)

        b = b.with_training_data_uri(conf.training_data_uri)
        b = b.with_training_output_uri(conf.training_output_uri)
        b = b.with_model_uri(conf.model_uri)

        return b.with_template(json_format.MessageToDict(conf.kc_config))

    def validate(self):
        super().validate()
        if not self.config.get('kc_config'):
            raise rv.ConfigError('You must specify a template for the backend '
                                 'configuration - use "with_template".')
        if self.require_task and not self.task:
            raise rv.ConfigError('You must specify the task this backend '
                                 'is for - use "with_task".')

    def build(self):
        """Build this configuration, setting any values into the
           TF object detection pipeline config as necessary.
        """
        self.validate()

        b = deepcopy(self)

        for config_mod, ignore_missing_keys, set_missing_keys in b.config_mods:
            try:
                set_nested_keys(b.config['kc_config'], config_mod,
                                ignore_missing_keys, set_missing_keys)
            except Exception as e:
                raise rv.ConfigError(
                    'Error setting configuration {}'.format(config_mod)) from e

        return KerasClassificationConfig(**b.config)

    def _applicable_tasks(self):
        return [rv.CHIP_CLASSIFICATION]

    def _process_task(self):
        return self.with_config(
            {
                'model': {
                    'inputSize': self.task.chip_size
                },
                'trainer': {
                    'options': {
                        'classNames': self.task.class_map.get_class_names(),
                        'inputSize': self.task.chip_size
                    }
                }
            },
            set_missing_keys=True)

    def _load_model_defaults(self, model_defaults):
        """Loads defaults. Expected keys are "pretrained_model_uri" and "pipeline_config_uri",
           neither of which is required.
        """
        expected_keys = ['pretrained_model_uri', 'kc_config']
        unknown_keys = set(model_defaults.keys()) - set(expected_keys)
        if unknown_keys:
            raise rv.ConfigError('Unexpected keys in model defaults:'
                                 ' {}. Expected keys: {}'.format(
                                     unknown_keys, expected_keys))

        b = self
        if 'pretrained_model_uri' in model_defaults:
            b = b.with_pretrained_model(model_defaults['pretrained_model_uri'])
        if 'kc_config' in model_defaults:
            b = b.with_template(model_defaults['kc_config'])
        return b

    def with_template(self, template):
        """Use a template from the dict, string or uri as the base for the
        Keras Classification API.
        """
        template_json = None
        if type(template) is dict:
            msg = json_format.ParseDict(template, PipelineConfig())

            template_json = json_format.MessageToDict(msg)
        else:
            # Try parsing the string as a message, on fail assume it's a URI
            msg = None
            try:
                msg = json_format.Parse(template, PipelineConfig())
            except json_format.ParseError:
                msg = json_format.Parse(
                    file_to_str(template), PipelineConfig())
            template_json = json_format.MessageToDict(msg)

        b = deepcopy(self)
        b.config['kc_config'] = template_json
        return b

    def with_batch_size(self, batch_size):
        """Sets the training batch size."""
        return self.with_config({
            'trainer': {
                'options': {
                    'batchSize': batch_size
                }
            }
        })

    def with_num_epochs(self, num_epochs):
        """Sets the number of training epochs."""
        return self.with_config({
            'trainer': {
                'options': {
                    'nbEpochs': num_epochs
                }
            }
        })

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

    def with_train_options(self,
                           sync_interval=600,
                           do_monitoring=True,
                           replace_model=False):
        """Sets the train options for this backend.

           Args:
              sync_interval: How often to sync output of training to
                             the cloud (in seconds).

              do_monitoring: Run process to monitor training (eg. Tensorboard)

              replace_model: Replace the model checkpoint if exists.
                             If false, this will continue training from
                             checkpoing if exists, if the backend allows for this.
        """
        b = deepcopy(self)
        b.config['train_options'] = KerasClassificationConfig.TrainOptions(
            sync_interval, do_monitoring, replace_model)

        return b
