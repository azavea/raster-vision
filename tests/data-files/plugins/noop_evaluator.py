import rastervision as rv
from rastervision.evaluation import (Evaluator, EvaluatorConfig,
                                     EvaluatorConfigBuilder)
from rastervision.protos.evaluator_pb2 import EvaluatorConfig as EvaluatorConfigMsg

from .noop_utils import noop

NOOP_EVALUATOR = 'NOOP_EVALUATOR'


class NoopEvaluator(Evaluator):
    def process(self, scenes, tmp_dir):
        return noop(scenes)


class NoopEvaluatorConfig(EvaluatorConfig):
    def __init__(self):
        super().__init__(NOOP_EVALUATOR)

    def to_proto(self):
        msg = EvaluatorConfigMsg(
            evaluator_type=self.evaluator_type, custom_config={})
        return msg

    def create_evaluator(self):
        return NoopEvaluator()

    def update_for_command(self, command_type, experiment_config, context=[]):
        return (self, rv.core.CommandIODefinition())


class NoopEvaluatorConfigBuilder(EvaluatorConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(NoopEvaluatorConfig, {})

    def from_proto(self, msg):
        return self


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.EVALUATOR, NOOP_EVALUATOR,
                                            NoopEvaluatorConfigBuilder)
