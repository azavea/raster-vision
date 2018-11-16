from unittest.mock import Mock

import rastervision as rv
from rastervision.analyzer import (Analyzer, AnalyzerConfig,
                                   AnalyzerConfigBuilder)  # noqa
from rastervision.protos.analyzer_pb2 \
    import AnalyzerConfig as AnalyzerConfigMsg # noqa

from tests.mock import SupressDeepCopyMixin

MOCK_ANALYZER = 'MOCK_ANALYZER'


class MockAnalyzer(Analyzer):
    def __init__(self):
        self.mock = Mock()

    def process(self, training_data, tmp_dir):
        self.mock.process(training_data, tmp_dir)


class MockAnalyzerConfig(SupressDeepCopyMixin, AnalyzerConfig):
    def __init__(self):
        super().__init__(MOCK_ANALYZER)
        self.mock = Mock()

        self.mock.to_proto.return_value = None
        self.mock.create_analyzer.return_value = None
        self.mock.update_for_command.return_value = None
        self.mock.save_bundle_files.return_value = (self, [])
        self.mock.load_bundle_files.return_value = self

    def to_proto(self):
        result = self.mock.to_proto()
        if result is None:
            return AnalyzerConfigMsg(analyzer_type=self.analyzer_type)
        else:
            return result

    def create_analyzer(self):
        result = self.mock.create_analyzer()
        if result is None:
            return MockAnalyzer()
        else:
            return result

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        result = self.mock.update_for_command(command_type, experiment_config,
                                              context, io_def)
        if result is None:
            return io_def or rv.core.CommandIODefinition()
        else:
            return result

    def save_bundle_files(self, bundle_dir):
        return self.mock.save_bundle_files(bundle_dir)

    def load_bundle_files(self, bundle_dir):
        return self.mock.load_bundle_files(bundle_dir)


class MockAnalyzerConfigBuilder(SupressDeepCopyMixin, AnalyzerConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(MockAnalyzerConfig, {})
        self.mock = Mock()
        self.mock.from_proto.return_value = None

    def from_proto(self, msg):
        result = self.mock.from_proto(msg)
        if result is None:
            return self
        else:
            return result
