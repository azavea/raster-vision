import unittest

from rastervision.core import CommandIODefinition


class TestCommandIoDefinition(unittest.TestCase):
    def test_add_missing(self):
        cid = CommandIODefinition()
        cid.add_missing('message')
        self.assertEqual(cid.missing_input_messages[0], 'message')
