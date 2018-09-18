from copy import deepcopy

import rastervision as rv
from rastervision.command import (TrainCommand, CommandConfig,
                                  CommandConfigBuilder)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg


class TrainCommandConfig(CommandConfig):
    def __init__(self, task, backend):
        super().__init__(rv.TRAIN)
        self.task = task
        self.backend = backend

    def create_command(self, tmp_dir):
        backend = self.backend.create_backend(self.task)
        task = self.task.create_task(backend)

        return TrainCommand(task)

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
        if prev is None:
            self.task = None
            self.backend = None
        else:
            self.task = prev.task
            self.backend = prev.backend

    def build(self):
        if self.task is None:
            raise rv.ConfigError(
                'Task not set. Use with_task or with_experiment')

        if self.backend is None:
            raise rv.ConfigError(
                'Backend not set. Use with_task or with_experiment')

        return TrainCommandConfig(self.task, self.backend)

    def from_proto(self, msg):
        self.process_plugins(msg)
        msg = msg.train_config
        task = rv.TaskConfig.from_proto(msg.task)
        backend = rv.BackendConfig.from_proto(msg.backend)

        b = self.with_task(task)
        b = b.with_backend(backend)

        return b

    def with_experiment(self, experiment_config):
        b = self.with_task(experiment_config.task)
        b = b.with_backend(experiment_config.backend)
        return b

    def with_task(self, task):
        b = deepcopy(self)
        b.task = task
        return b

    def with_backend(self, backend):
        b = deepcopy(self)
        b.backend = backend
        return b
