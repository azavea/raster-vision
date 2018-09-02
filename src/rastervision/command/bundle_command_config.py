from copy import deepcopy

import rastervision as rv
from rastervision.command import (BundleCommand,
                                  CommandConfig,
                                  CommandConfigBuilder)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg

class PredictCommandConfig(CommandConfig):
    def __init__(self,
                 task,
                 backend,
                 scene,
                 analyzers):
        super().__init__(rv.BUNDLE)
        self.task = task
        self.backend = backend
        self.scene = scene
        self.analyzers = analyzers

    def create_command(self, tmp_dir):
        return BundleCommand(task, backend, scene, analyzers)

    def to_proto(self):
        msg = super().to_proto()

        task = self.task.to_proto()
        backend = self.backend.to_proto()
        scene = self.scene.to_proto()
        analyzers = list(map(lambda a: a.to_proto(), self.analyzers))

        msg.MergeFrom(CommandConfigMsg(
            bundle_config=CommandConfigMsg.BundleConfig(task=task,
                                                        backend=backend,
                                                        scene=scene)))

        return msg

    @staticmethod
    def builder():
        return BundleCommandConfigBuilder()

class BundleCommandConfigBuilder(CommandConfigBuilder):
    def __init__(self):
        self.task = None
        self.backend = None
        self.scene = None
        self.analyzers = []

    def build(self):
        if self.task is None:
            raise rv.ConfigError("Task not set. Use with_task or with_experiment")

        if self.backend is None:
            raise rv.ConfigError("Backend not set. Use with_backend or with_experiment")

        if self.scene is None:
            raise rv.ConfigError("Template scene not set. Use with_scene or with_experiment")

        return PredictCommandConfig(self.task,
                                    self.backend,
                                    self.scene,
                                    self.analyzers)


    def from_proto(self, msg):
        msg = msg.predict_config

        task = rv.TaskConfig.from_proto(msg.task)
        backend = rv.BackendConfig.from_proto(msg.backend)
        scene = rv.SceneConfig.from_proto(msg.scene)
        analyzers = list(map(rv.AnalyzerConfig.from_proto,
                             msg.analyzers))

        b = self.with_task(task)
        b = b.with_backend(backend)
        b = b.with_scene(scene)
        b = b.with_analyzers(analyzers)

        return b

    def with_experiment(self, experiment_config):
        b = self.with_task(experiment_config.task)
        b = b.with_backend(experiment_config.backend)
        b = b.with_scene(experiment_config.dataset.all_scenes()[0])
        b = b.with_analyzers(experiment_config.analyzers)
        return  b

    def with_task(self, task):
        b = deepcopy(self)
        b.task = task
        return b

    def with_backend(self, backend):
        b = deepcopy(self)
        b.backend = backend
        return b


    def with_scene(self, scene):
        b = deepcopy(self)
        b.scene = scene
        return b

    def with_analyzers(self, analyzers):
        b = deepcopy(self)
        b.analyzers = analyzers
        return b
