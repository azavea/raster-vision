from copy import deepcopy

import rastervision as rv
from rastervision.command import (CommandConfig, CommandConfigBuilder,
                                  BundleCommand)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg
from rastervision.rv_config import RVConfig


class BundleCommandConfig(CommandConfig):
    def __init__(self, root_uri, task, backend, scene, analyzers):
        super().__init__(rv.BUNDLE, root_uri)
        self.task = task
        self.backend = backend
        self.scene = scene
        self.analyzers = analyzers

    def create_command(self, tmp_dir=None):
        if not tmp_dir:
            _tmp_dir = RVConfig.get_tmp_dir()
            tmp_dir = _tmp_dir.name
        else:
            _tmp_dir = tmp_dir

        retval = BundleCommand(self, self.task, self.backend, self.scene,
                               self.analyzers)
        retval.set_tmp_dir(_tmp_dir)
        return retval

    def to_proto(self):
        msg = super().to_proto()

        task = self.task.to_proto()
        backend = self.backend.to_proto()
        scene = self.scene.to_proto()
        analyzers = list(map(lambda a: a.to_proto(), self.analyzers))

        b = CommandConfigMsg.BundleConfig(
            task=task, backend=backend, scene=scene, analyzers=analyzers)

        msg.MergeFrom(CommandConfigMsg(bundle_config=b))

        return msg

    @staticmethod
    def builder():
        return BundleCommandConfigBuilder()


class BundleCommandConfigBuilder(CommandConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(prev)
        if prev is None:
            self.task = None
            self.backend = None
            self.scene = None
            self.analyzers = None
        else:
            self.task = prev.task
            self.backend = prev.backend
            self.scene = prev.scene
            self.analyzers = prev.analyzers

    def validate(self):
        super().validate()
        if self.task is None:
            raise rv.ConfigError('Task not set for BundleCommandConfig. '
                                 'Use with_task or with_experiment')
        if self.backend is None:
            raise rv.ConfigError('Backend not set for BundleCommandConfig. '
                                 'Use with_backend or with_experiment')
        if self.scene is None:
            raise rv.ConfigError(
                'Template scene not set for BundleCommandConfig. '
                'Use with_scene or with_experiment')
        if self.analyzers is None:
            raise rv.ConfigError('Analyzers not set for BundleCommandConfig. '
                                 'Use with_analyzers or with_experiment')

    def build(self):
        self.validate()
        return BundleCommandConfig(self.root_uri, self.task, self.backend,
                                   self.scene, self.analyzers)

    def from_proto(self, msg):
        b = super().from_proto(msg)

        conf = msg.bundle_config

        task = rv.TaskConfig.from_proto(conf.task)
        backend = rv.BackendConfig.from_proto(conf.backend)
        scene = rv.SceneConfig.from_proto(conf.scene)
        analyzers = list(map(rv.AnalyzerConfig.from_proto, conf.analyzers))

        b = b.with_task(task)
        b = b.with_backend(backend)
        b = b.with_scene(scene)
        b = b.with_analyzers(analyzers)

        return b

    def get_root_uri(self, experiment_config):
        return experiment_config.bundle_uri

    def with_experiment(self, experiment_config):
        b = super().with_experiment(experiment_config)
        b = b.with_task(experiment_config.task)
        b = b.with_backend(experiment_config.backend)
        b = b.with_scene(experiment_config.dataset.all_scenes()[0])
        b = b.with_analyzers(experiment_config.analyzers)
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

    def with_scene(self, scene):
        b = deepcopy(self)
        b.scene = scene
        return b

    def with_analyzers(self, analyzers):
        b = deepcopy(self)
        b.analyzers = analyzers
        return b
