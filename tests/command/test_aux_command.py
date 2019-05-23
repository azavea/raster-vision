import unittest
from functools import reduce

import rastervision as rv
from rastervision.rv_config import RVConfig
from rastervision.runner import CommandDefinition

import tests.mock as mk


class TestAuxCommand(mk.MockMixin, unittest.TestCase):
    def test_command_create(self):
        with RVConfig.get_tmp_dir() as tmp_dir:
            uris = [('one', '1'), ('two', '2')]
            cmd_conf = rv.CommandConfig.builder(mk.MOCK_AUX_COMMAND) \
                                       .with_config(uris=uris) \
                                       .with_root_uri(tmp_dir) \
                                       .build()

            cmd_conf = rv.command.CommandConfig.from_proto(cmd_conf.to_proto())
            cmd = cmd_conf.create_command()

            self.assertTrue(cmd, mk.MockAuxCommand)

            cmd.run(tmp_dir)

            self.assertTrue(cmd.mock.run.called)

    def test_command_from_experiment(self):
        with RVConfig.get_tmp_dir() as tmp_dir:
            uris = [('one', '1'), ('two', '2'), ('three', '3'), ('four', '4')]

            e = mk.create_mock_experiment().to_builder() \
                                           .with_root_uri(tmp_dir) \
                                           .with_custom_config({
                                               'mock_aux_command': {
                                                   'key': 'mock',
                                                   'config': {
                                                       'uris': uris
                                                   }
                                               }
                                           }) \
                                           .build()

            rv.ExperimentRunner.get_runner(rv.LOCAL).run(
                e, splits=2, commands_to_run=[mk.MOCK_AUX_COMMAND])

            # Nothing to assert here, just ensures code path runs.

    def test_command_split(self):
        with RVConfig.get_tmp_dir() as tmp_dir:
            uris = [('one', '1'), ('two', '2'), ('three', '3'), ('four', '4')]

            cmd_conf = rv.CommandConfig.builder(mk.MOCK_AUX_COMMAND) \
                                       .with_config(uris=uris) \
                                       .with_root_uri(tmp_dir) \
                                       .build()

            defs = CommandDefinition.from_command_configs(
                [cmd_conf], [mk.MOCK_AUX_COMMAND], 2)

            self.assertEqual(len(defs), 2)

            outputs = reduce(lambda a, b: a.union(b),
                             map(lambda x: x.io_def.output_uris, defs))

            self.assertEqual(outputs, set(['1', '2', '3', '4']))

    def test_required_fields(self):
        with RVConfig.get_tmp_dir() as tmp_dir:
            b = rv.CommandConfig.builder(mk.MOCK_AUX_COMMAND) \
                                .with_config() \
                                .with_root_uri(tmp_dir)
            with self.assertRaises(rv.ConfigError) as context:
                b.build()

            self.assertTrue('uris' in str(context.exception))


if __name__ == '__main__':
    unittest.main()
