from unittest.mock import Mock

import rastervision as rv

MOCK_AUX_COMMAND = 'MOCK_AUX_COMMAND'


class MockAuxCommand(rv.AuxCommand):
    command_type = MOCK_AUX_COMMAND
    options = rv.AuxCommandOptions(
        split_on='uris',
        outputs=lambda conf: list(map(lambda x: x[1], conf['uris'])),
        required_fields=['uris'])
    mock = Mock()

    def run(self, tmp_dir=None):
        self.mock.run(tmp_dir)
