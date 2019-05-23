import os
from copy import deepcopy
import json

from google.protobuf import struct_pb2

import rastervision as rv
from rastervision.rv_config import RVConfig
from rastervision.command import (CommandConfig, CommandConfigBuilder)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg
from rastervision.utils.misc import split_into_groups


class AuxCommandConfig(CommandConfig):
    def __init__(self, command_class, root_uri, config):
        super().__init__(command_class.command_type, root_uri)
        self.command_class = command_class
        self.command_options = command_class.options
        self.config = config

    def create_command(self, tmp_dir=None):
        if not tmp_dir:
            _tmp_dir = RVConfig.get_tmp_dir()
            tmp_dir = _tmp_dir.name
        else:
            _tmp_dir = tmp_dir

        retval = self.command_class(self.config)
        retval.set_tmp_dir(_tmp_dir)

        return retval

    def to_proto(self):
        msg = super().to_proto()
        conf = struct_pb2.Struct()
        conf['json'] = json.dumps(self.config)
        msg.MergeFrom(CommandConfigMsg(custom_config=conf))

        return msg

    def report_io(self):
        io_def = rv.core.CommandIODefinition()
        inputs = self.command_options.inputs(self.config)
        outputs = self.command_options.outputs(self.config)

        if inputs:
            io_def.add_inputs(inputs)
        if outputs:
            io_def.add_outputs(outputs)

        return io_def

    def split(self, num_parts):
        split_on = self.command_options.split_on
        if split_on:
            commands = []
            for i, split_elements in enumerate(
                    split_into_groups(self.config[split_on], num_parts)):
                split_config = deepcopy(self.config)
                split_config[split_on] = split_elements
                c = self.to_builder() \
                        .with_config(**split_config) \
                        .with_split_id(i) \
                        .build()
                commands.append(c)
            return commands
        else:
            return [self]


class AuxCommandConfigBuilder(CommandConfigBuilder):
    def __init__(self, command_type, prev=None):
        super().__init__(command_type, prev)
        if prev:
            self.config = prev.config
            self.command_class = prev.command_class
        else:
            self.config = None
            self.command_class = None

    def validate(self):
        super().validate()

        if not self.command_class:
            raise rv.ConfigError(
                'AuxCommandConfigBuilder requires the command_class be set.')

        if self.config is None:
            raise rv.ConfigError(
                'AuxCommandConfigBuilder requires a configuration be set, either '
                'through with_config or by setting a dict with the property "config" '
                'in an experiment custom configuration dict with the command name '
                'as the key in the experiment custom configuration')

        if self.command_class.options.required_fields:
            for field in self.command_class.options.required_fields:
                if field not in self.config:
                    raise rv.ConfigError('{} command requires the field {} '
                                         'be set in the configuration.'.format(
                                             self.command_type, field))

    def build(self):
        self.validate()
        return AuxCommandConfig(self.command_class, self.root_uri, self.config)

    def from_proto(self, msg):
        b = super().from_proto(msg)

        self.command_class = rv._registry.get_aux_command_class(
            self.command_type)

        b = b.with_config(**json.loads(msg.custom_config['json']))

        return b

    def get_root_uri(self, experiment_config):
        if self.root_uri:
            return self.root_uri
        command_name = self.command_type.lower()
        command_config = experiment_config.custom_config.get(
            '{}'.format(command_name))
        if not command_config:
            raise rv.ConfigError(
                '{} command requires experiment custom_config '
                'contains a {} key'.format(self.command_type, command_name))
        key = command_config.get('key')
        if not key:
            root_uri = command_config.get('root_uri')
            if not root_uri:
                raise rv.ConfigError(
                    '{} command requires a "key" or "root_uri" '
                    'be set in the command config dict inside the '
                    'experiment custom_config'.format(self.command_type))
        else:
            root_uri = os.path.join(experiment_config.root_uri, command_name,
                                    key)

        return root_uri

    def with_experiment(self, experiment_config):
        b = super().with_experiment(experiment_config)
        command_name = self.command_type.lower()
        command_config = experiment_config.custom_config.get(
            '{}'.format(command_name))
        if command_config:
            config_data = command_config.get('config')
            if not config_data:
                raise rv.ConfigError(
                    '{} command requires a configuration dict set in the '
                    '"config" property of the experiment '
                    'custom_config for this command.'.format(
                        self.command_type))
            else:
                b = b.with_config(**config_data)

        return b

    def with_config(self, **kwargs):
        b = deepcopy(self)
        b.config = kwargs
        return b

    def with_command_class(self, command_class):
        b = deepcopy(self)
        b.command_class = command_class
        return b
