from copy import deepcopy

import rastervision as rv
from rastervision.command import (EvalCommand,
                                  CommandConfig,
                                  CommandConfigBuilder,
                                  NoOpCommand)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg

class EvalCommandConfig(CommandConfig):
    def __init__(self,
                 task,
                 scenes,
                 evaluators):
        super().__init__(rv.EVAL)
        self.task = task
        self.scenes = scenes
        self.evaluators = evaluators

    def create_command(self, tmp_dir):
        if len(self.scenes) == 0 or len(self.evaluators) == 0:
            return NoOpCommand()

        scenes = list(map(lambda s: s.create_scene(self.task, tmp_dir),
                          self.scenes))
        evaluators = list(map(lambda a: a.create_evaluator(),
                             self.evaluators))
        return EvalCommand(scenes, evaluators)

    def to_proto(self):
        msg = super().to_proto()
        task = self.task.to_proto()
        scenes = list(map(lambda s: s.to_proto(), self.scenes))
        evaluators = list(map(lambda e: e.to_proto(), self.evaluators))

        msg.MergeFrom(CommandConfigMsg(
            eval_config=CommandConfigMsg.EvalConfig(task=task,
                                                    scenes=scenes,
                                                    evaluators=evaluators)))

        return msg

    @staticmethod
    def builder():
        return EvalCommandConfigBuilder()

class EvalCommandConfigBuilder(CommandConfigBuilder):
    def __init__(self):
        self.task = None
        self.scenes = None
        self.evaluators = None

    def build(self):
        if self.task is None:
            raise rv.ConfigError("task not set. Use with_task or with_experiment")
        if self.scenes is None:
            raise rv.ConfigError("scenes not set. Use with_scenes or with_experiment")
        if self.evaluators is None:
            raise rv.ConfigError("evaluators not set. Use with_evaluators or with_experiment")
        return EvalCommandConfig(self.task,
                                 self.scenes,
                                 self.evaluators)


    def from_proto(self, msg):
        msg = msg.eval_config

        task = rv.TaskConfig.from_proto(msg.task)
        scenes = list(map(rv.SceneConfig.from_proto,
                          msg.scenes))
        evaluators = list(map(rv.EvaluatorConfig.from_proto,
                              msg.evaluators))

        b = self.with_task(task)
        b = b.with_scenes(scenes)
        b = b.with_evaluators(evaluators)

        return b

    def with_experiment(self, experiment_config):
        b = self.with_task(experiment_config.task)
        b = b.with_scenes(experiment_config.dataset.validation_scenes)
        b = b.with_evaluators(experiment_config.evaluators)
        return  b

    def with_task(self, task):
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
