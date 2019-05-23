import os
import unittest

import rasterio

import rastervision as rv
from rastervision.rv_config import RVConfig
from tests import data_file_path
import tests.mock as mk


class TestCogifyCommand(mk.MockMixin, unittest.TestCase):
    def test_command_create(self):
        src_path = data_file_path('small-rgb-tile.tif')
        with RVConfig.get_tmp_dir() as tmp_dir:
            cog_path = os.path.join(tmp_dir, 'cog.tif')

            cmd_conf = rv.CommandConfig.builder(rv.COGIFY) \
                                       .with_root_uri(tmp_dir) \
                                       .with_config(uris=[(src_path, cog_path)],
                                                    block_size=128) \
                                       .build()

            cmd_conf = rv.command.CommandConfig.from_proto(cmd_conf.to_proto())
            cmd = cmd_conf.create_command()

            self.assertTrue(cmd, rv.command.aux.CogifyCommand)

            cmd.run(tmp_dir)

            # Check that it's cogified
            with rasterio.open(cog_path) as ds:
                self.assertEqual(ds.block_shapes, [(128, 128), (128, 128),
                                                   (128, 128)])
                self.assertEqual(ds.overviews(1), [2, 4, 8, 16, 32])

    def test_command_through_experiment(self):
        src_path = data_file_path('small-rgb-tile.tif')
        with RVConfig.get_tmp_dir() as tmp_dir:
            cog_path = os.path.join(tmp_dir, 'cog.tif')

            e = mk.create_mock_experiment().to_builder() \
                                           .with_root_uri(tmp_dir) \
                                           .with_custom_config({
                                               'cogify': {
                                                   'key': 'test',
                                                   'config': {
                                                       'uris': [(src_path, cog_path)],
                                                       'block_size': 128
                                                   }
                                               }
                                           }) \
                                           .build()

            rv.ExperimentRunner.get_runner(rv.LOCAL).run(
                e, splits=2, commands_to_run=[rv.COGIFY])

            # Check that it's cogified
            with rasterio.open(cog_path) as ds:
                self.assertEqual(ds.block_shapes, [(128, 128), (128, 128),
                                                   (128, 128)])
                self.assertEqual(ds.overviews(1), [2, 4, 8, 16, 32])


if __name__ == '__main__':
    unittest.main()
