from abc import abstractmethod

import rastervision as rv
from rastervision.core import (Config, ConfigBuilder)


class EvaluatorConfig(Config):
    def __init__(self, evaluator_type):
        self.evaluator_type = evaluator_type

    @abstractmethod
    def create_evaluator(self):
        """Create the Evaluator that this configuration represents"""
        pass

    def to_builder(self):
        return rv._registry.get_config_builder(rv.EVALUATOR,
                                               self.evaluator_type)(self)

    @staticmethod
    def builder(evaluator_type):
        return rv._registry.get_config_builder(rv.EVALUATOR, evaluator_type)()

    @staticmethod
    def from_proto(msg):
        """Creates a EvaluatorConfig from the specificed protobuf message
        """
        return rv._registry.get_config_builder(rv.EVALUATOR, msg.evaluator_type)() \
                           .from_proto(msg) \
                           .build()


class EvaluatorConfigBuilder(ConfigBuilder):
    pass
