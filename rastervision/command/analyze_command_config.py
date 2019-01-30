from copy import deepcopy

import rastervision as rv
from rastervision.command import (AnalyzeCommand, CommandConfig,
                                  CommandConfigBuilder, NoOpCommand)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg
from rastervision.command.utils import (check_scenes_type,
                                        check_analyzers_type)


class AnalyzeCommandConfig(CommandConfig):
    def __init__(self, root_uri, task, scenes, analyzers):
        super().__init__(rv.ANALYZE, root_uri)
        self.task = task
        self.scenes = scenes
        self.analyzers = analyzers

    def create_command(self, tmp_dir=None):
        if len(self.scenes) == 0 or len(self.analyzers) == 0:
            return NoOpCommand()

        retval = AnalyzeCommand(self)
        retval.set_tmp_dir(tmp_dir)
        return retval

    def to_proto(self):
        msg = super().to_proto()
        task = self.task.to_proto()
        scenes = list(map(lambda s: s.to_proto(), self.scenes))
        analyzers = list(map(lambda a: a.to_proto(), self.analyzers))

        msg.MergeFrom(
            CommandConfigMsg(
                analyze_config=CommandConfigMsg.AnalyzeConfig(
                    task=task, scenes=scenes, analyzers=analyzers)))

        return msg

    def report_io(self):
        io_def = rv.core.CommandIODefinition()
        self.task.report_io(self.command_type, io_def)
        for scene in self.scenes:
            scene.report_io(self.command_type, io_def)
        for analyzer in self.analyzers:
            analyzer.report_io(self.command_type, io_def)
        return io_def

    @staticmethod
    def builder():
        return AnalyzeCommandConfigBuilder()


class AnalyzeCommandConfigBuilder(CommandConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(prev)
        if prev is None:
            self.task = None
            self.scenes = None
            self.analyzers = None
        else:
            self.task = prev.task
            self.scenes = prev.scenes
            self.analyzers = prev.analyzers

    def validate(self):
        super().validate()
        if self.scenes is None:
            raise rv.ConfigError('scenes not set for AnalyzeCommandConfig. Use'
                                 ' with_scenes or with_experiment')
        check_scenes_type(self.scenes)
        if self.analyzers is None:
            raise rv.ConfigError(
                'analyzers not set. Use with_analyzers or with_experiment')
        check_analyzers_type(self.analyzers)

    def build(self):
        self.validate()
        return AnalyzeCommandConfig(self.root_uri, self.task, self.scenes,
                                    self.analyzers)

    def from_proto(self, msg):
        b = super().from_proto(msg)

        conf = msg.analyze_config

        task = rv.TaskConfig.from_proto(conf.task)
        scenes = list(map(rv.SceneConfig.from_proto, conf.scenes))
        analyzers = list(map(rv.AnalyzerConfig.from_proto, conf.analyzers))

        b = b.with_task(task)
        b = b.with_scenes(scenes)
        b = b.with_analyzers(analyzers)

        return b

    def get_root_uri(self, experiment_config):
        return experiment_config.analyze_uri

    def with_experiment(self, experiment_config):
        b = super().with_experiment(experiment_config)
        b = b.with_task(experiment_config.task)
        b = b.with_scenes(experiment_config.dataset.all_scenes())
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

    def with_scenes(self, scenes):
        b = deepcopy(self)
        b.scenes = scenes
        return b

    def with_analyzers(self, analyzers):
        b = deepcopy(self)
        b.analyzers = analyzers
        return b
