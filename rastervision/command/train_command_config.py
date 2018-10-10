from copy import deepcopy

import rastervision as rv
from rastervision.command import (TrainCommand, CommandConfig,
                                  CommandConfigBuilder)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg
from rastervision.rv_config import RVConfig


class TrainCommandConfig(CommandConfig):
    def __init__(self, root_uri, task, backend):
        super().__init__(rv.TRAIN, root_uri)
        self.task = task
        self.backend = backend

    def create_command(self, tmp_dir=None):
        if not tmp_dir:
            _tmp_dir = RVConfig.get_tmp_dir()
            tmp_dir = _tmp_dir.name
        else:
            _tmp_dir = tmp_dir

        backend = self.backend.create_backend(self.task)
        task = self.task.create_task(backend)
        retval = TrainCommand(task)
        retval.set_tmp_dir(_tmp_dir)
        return retval

    def to_proto(self):
        msg = super().to_proto()

        task = self.task.to_proto()
        backend = self.backend.to_proto()

        msg.MergeFrom(
            CommandConfigMsg(
                train_config=CommandConfigMsg.TrainConfig(
                    task=task, backend=backend)))

        return msg

    @staticmethod
    def builder():
        return TrainCommandConfigBuilder()


class TrainCommandConfigBuilder(CommandConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(prev)
        if prev is None:
            self.task = None
            self.backend = None
        else:
            self.task = prev.task
            self.backend = prev.backend

    def validate(self):
        super().validate()
        if self.task is None:
            raise rv.ConfigError('Task not set for TrainCommandConfig. Use '
                                 'with_task or with_experiment')

        if self.backend is None:
            raise rv.ConfigError('Backend not set for TrainCommandConfig. Use '
                                 'with_task or with_experiment')

    def build(self):
        self.validate()
        return TrainCommandConfig(self.root_uri, self.task, self.backend)

    def from_proto(self, msg):
        b = super().from_proto(msg)

        conf = msg.train_config

        task = rv.TaskConfig.from_proto(conf.task)
        backend = rv.BackendConfig.from_proto(conf.backend)

        b = b.with_task(task)
        b = b.with_backend(backend)

        return b

    def get_root_uri(self, experiment_config):
        return experiment_config.train_uri

    def with_experiment(self, experiment_config):
        b = super().with_experiment(experiment_config)
        b = b.with_task(experiment_config.task)
        b = b.with_backend(experiment_config.backend)
        return b

    def with_task(self, task):
        """Sets a specific task type.

        Args:
            task:  A TaskConfig object.

        """
        b = deepcopy(self)
        b.task = task
        return b

    def with_backend(self, backend):
        b = deepcopy(self)
        b.backend = backend
        return b
