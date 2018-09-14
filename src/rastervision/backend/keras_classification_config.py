import os
from copy import deepcopy
from google.protobuf import (json_format)

import rastervision as rv
from rastervision.backend import (BackendConfig, BackendConfigBuilder,
                                  KerasClassification)
from rastervision.core.config import set_nested_keys
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg
from rastervision.utils.files import file_to_str

# Default location to Tensorflow Object Detection's scripts.
CHIP_OUTPUT_FILES = ["training.zip", "validation.zip"]


class KerasClassificationConfig(BackendConfig):
    class TrainOptions:
        def __init__(self, sync_interval=600, replace_model=True):
            self.sync_interval = sync_interval
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
        return KerasClassification(self, task_config)

    def to_proto(self):
        d = {
            "sync_interval": self.train_options.sync_interval,
            "replace_model": self.train_options.replace_model,
            "kc_config": self.kc_config
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

    def preprocess_command(self, command_type, experiment_config,
                           context=None):
        conf = self
        io_def = rv.core.CommandIODefinition()
        if command_type == rv.CHIP:
            conf.training_data_uri = experiment_config.chip_uri

            outputs = list(
                map(lambda x: os.path.join(conf.training_data_uri, x),
                    CHIP_OUTPUT_FILES))
            io_def.add_outputs(outputs)
        if command_type == rv.TRAIN:
            conf.training_output_uri = experiment_config.train_uri
            inputs = list(
                map(lambda x: os.path.join(experiment_config.chip_uri, x),
                    CHIP_OUTPUT_FILES))
            io_def.add_inputs(inputs)

            # TODO: Change? Or make configurable?
            conf.model_uri = os.path.join(conf.training_output_uri, "model")
            io_def.add_output(conf.model_uri)
        if command_type == rv.PREDICT:
            io_def.add_input(conf.model_uri)

        return (conf, io_def)


class KerasClassificationConfigBuilder(BackendConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                "kc_config": prev.kc_config,
                "pretrained_model_uri": prev.pretrained_model_uri,
                "train_options": prev.train_options,
                "debug": prev.debug,
                "training_data_uri": prev.trainind_data_uri,
                "training_output_uri": prev.training_output_uri,
                "model_uri": prev.model_uri
            }
        super().__init__(rv.KERAS_CLASSIFICATION, KerasClassificationConfig,
                         config, prev)
        self.config_mods = []
        self.require_task = True

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
            sync_interval=conf.sync_interval, replace_model=conf.replace_model)
        # TODO: Debug
        # b = b.with_debug(conf.debug)

        return b.with_template(json_format.MessageToDict(conf.kc_config))

    def validate(self):
        super().validate()
        if not self.config.get('kc_config'):
            raise rv.ConfigError("You must specify a template for the backend "
                                 "configuration - use 'with_template'.")
        if self.require_task and not self.task:
            raise rv.ConfigError("You must specify the task this backend "
                                 "is for - use 'with_task'.")
        return True

    def build(self):
        """Build this configuration, setting any values into the
           TF object detection pipeline config as necessary.
        """
        self.validate()

        b = deepcopy(self)

        for config_mod, ignore_missing_keys, set_missing_keys in b.config_mods:
            set_nested_keys(b.config['kc_config'], config_mod,
                            ignore_missing_keys, set_missing_keys)

        return KerasClassificationConfig(**b.config)

    def _applicable_tasks(self):
        return [rv.CHIP_CLASSIFICATION]

    def _process_task(self):
        return self.with_config(
            {
                "trainer": {
                    "options": {
                        "classNames": self.task.class_map.get_class_names(),
                    }
                }
            },
            set_missing_keys=True)

    def _load_model_defaults(self, model_defaults):
        """Loads defaults. Expected keys are "pretrained_model_uri" and "pipeline_config_uri",
           neither of which is required.
        """
        expected_keys = ["pretrained_model_uri", "kc_config"]
        unknown_keys = set(model_defaults.keys()) - set(expected_keys)
        if unknown_keys:
            raise rv.ConfigError("Unexpected keys in model defaults:"
                                 " {}. Expected keys: {}".format(
                                     unknown_keys, expected_keys))

        b = self
        if "pretrained_model_uri" in model_defaults:
            b = b.with_pretrained_model(model_defaults["pretrained_model_uri"])
        if "kc_config" in model_defaults:
            b = b.with_template(model_defaults["kc_config"])
        return b

    def with_template(self, template):
        """Use a template from the dict, string or uri as the base for the
        Keras Classification API.
        """
        from keras_classification.protos.pipeline_pb2 import PipelineConfig

        template_json = None
        if type(template) is dict:
            # import  pdb; pdb.set_trace()
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
        return self.with_config({
            "trainer": {
                "options": {
                    "batchSize": batch_size
                }
            }
        })

    def with_num_epochs(self, num_epochs):
        return self.with_config({
            "trainer": {
                "options": {
                    "nb_epochs": num_epochs
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

    def with_train_options(self, sync_interval=600, replace_model=False):
        """Sets the train options for this backend.

           Args:
              sync_interval: How often to sync output of training to
                             the cloud (in seconds).

              replace_model: Replace the model checkpoint if exists.
                             If false, this will continue training from
                             checkpoing if exists, if the backend allows for this.
        """
        b = deepcopy(self)
        b.config['train_options'] = KerasClassificationConfig.TrainOptions(
            sync_interval, replace_model)
        return b
