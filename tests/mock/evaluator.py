import unittest
from unittest.mock import Mock

import rastervision as rv
from rastervision.evaluation import (Evaluator, EvaluatorConfig,
                                     EvaluatorConfigBuilder)
from rastervision.protos.evaluator_pb2 \
    import EvaluatorConfig as EvaluatorConfigMsg

from tests.mock import SupressDeepCopyMixin

MOCK_EVALUATOR = 'MOCK_EVALUATOR'


class MockEvaluator(Evaluator):
    def __init__(self):
        self.mock = Mock()

    def process(self, scenes, tmp_dir):
        self.mock.process(scenes, tmp_dir)


class MockEvaluatorConfig(SupressDeepCopyMixin, EvaluatorConfig):
    def __init__(self):
        super().__init__(MOCK_EVALUATOR)
        self.mock = Mock()

        self.mock.to_proto.return_value = None
        self.mock.create_evaluator.return_value = None
        self.mock.update_for_command.return_value = None

    def to_proto(self):
        result = self.mock.to_proto()
        if result is None:
            return EvaluatorConfigMsg(evaluator_type=self.evaluator_type)
        else:
            return result

    def create_evaluator(self):
        result = self.mock.create_evaluator()
        if result is None:
            return MockEvaluator()
        else:
            return result

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None):
        super().update_for_command(command_type, experiment_config, context)
        self.mock.update_for_command(command_type, experiment_config, context)

    def report_io(self, command_type, io_def):
        self.mock.report_io(command_type, io_def)


class MockEvaluatorConfigBuilder(SupressDeepCopyMixin, EvaluatorConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(MockEvaluatorConfig, {})
        self.mock = Mock()
        self.mock.from_proto.return_value = None

    def from_proto(self, msg):
        result = self.mock.from_proto(msg)
        if result is None:
            return self
        else:
            return result


if __name__ == '__main__':
    unittest.main()
