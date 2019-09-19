from copy import deepcopy

import rastervision as rv
from rastervision.command import (EvalCommand, CommandConfig,
                                  CommandConfigBuilder, NoOpCommand)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg
from rastervision.rv_config import RVConfig
from rastervision.command.utils import (check_scenes_type, check_task_type)
from rastervision.evaluation import EvaluatorConfig


class EvalCommandConfig(CommandConfig):
    def __init__(self, root_uri, task, scenes, evaluators):
        super().__init__(rv.EVAL, root_uri)
        self.task = task
        self.scenes = scenes
        self.evaluators = evaluators

    def create_command(self, tmp_dir=None):
        if len(self.scenes) == 0 or len(self.evaluators) == 0:
            return NoOpCommand()

        if not tmp_dir:
            _tmp_dir = RVConfig.get_tmp_dir()
            tmp_dir = _tmp_dir.name
        else:
            _tmp_dir = tmp_dir

        retval = EvalCommand(self)
        retval.set_tmp_dir(_tmp_dir)
        return retval

    def to_proto(self):
        msg = super().to_proto()
        task = self.task.to_proto()
        scenes = list(map(lambda s: s.to_proto(), self.scenes))
        evaluators = list(map(lambda e: e.to_proto(), self.evaluators))

        msg.MergeFrom(
            CommandConfigMsg(eval_config=CommandConfigMsg.EvalConfig(
                task=task, scenes=scenes, evaluators=evaluators)))

        return msg

    def report_io(self):
        io_def = rv.core.CommandIODefinition()
        self.task.report_io(self.command_type, io_def)
        for scene in self.scenes:
            scene.report_io(self.command_type, io_def)
        for evaluator in self.evaluators:
            evaluator.report_io(self.command_type, io_def)
        return io_def


class EvalCommandConfigBuilder(CommandConfigBuilder):
    def __init__(self, command_type, prev=None):
        super().__init__(command_type, prev)
        if prev is None:
            self.task = None
            self.scenes = None
            self.evaluators = None
        else:
            self.task = prev.task
            self.scenes = prev.scenes
            self.evaluators = prev.evaluators

    def validate(self):
        super().validate()
        if self.task is None:
            raise rv.ConfigError(
                'task not set for EvalCommandConfig. Use with_task or '
                'with_experiment')
        check_task_type(self.task)
        if self.scenes is None:
            raise rv.ConfigError(
                'scenes not set for EvalCommandConfig. Use with_scenes or '
                'with_experiment')
        check_scenes_type(self.scenes)
        if self.evaluators is None:
            raise rv.ConfigError(
                'evaluators not set. Use with_evaluators or with_experiment')
        if not isinstance(self.evaluators, list):
            raise rv.ConfigError(
                'evaluators must be a list of EvaluatorConfig objects, got {}'.
                format(type(self.evaluators)))
        for evaluator in self.evaluators:
            if not issubclass(type(evaluator), EvaluatorConfig):
                if not isinstance(evaluator, str):
                    raise rv.ConfigError(
                        'evaluators must be a subclass of EvaluatorConfig or string,'
                        ' got {}'.format(type(evaluator)))

    def build(self):
        self.validate()
        return EvalCommandConfig(self.root_uri, self.task, self.scenes,
                                 self.evaluators)

    def from_proto(self, msg):
        b = super().from_proto(msg)

        conf = msg.eval_config

        task = rv.TaskConfig.from_proto(conf.task)
        scenes = list(map(rv.SceneConfig.from_proto, conf.scenes))
        evaluators = list(map(rv.EvaluatorConfig.from_proto, conf.evaluators))

        b = b.with_task(task)
        b = b.with_scenes(scenes)
        b = b.with_evaluators(evaluators)

        return b

    def get_root_uri(self, experiment_config):
        return experiment_config.eval_uri

    def with_experiment(self, experiment_config):
        b = super().with_experiment(experiment_config)
        b = b.with_task(experiment_config.task)
        b = b.with_scenes(experiment_config.dataset.validation_scenes)
        b = b.with_evaluators(experiment_config.evaluators)
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

    def with_evaluators(self, evaluators):
        b = deepcopy(self)
        b.evaluators = evaluators
        return b
