import unittest
from unittest.mock import Mock
import numpy as np

from rastervision.core import Box
from rastervision.data import (RasterSource, RasterSourceConfig,
                               RasterSourceConfigBuilder,
                               IdentityCRSTransformer)
from rastervision.data import (ActivateMixin)
from rastervision.protos.raster_source_pb2 \
    import RasterSourceConfig as RasterSourceConfigMsg

from tests.mock import SupressDeepCopyMixin

MOCK_SOURCE = 'MOCK_SOURCE'


class MockRasterSource(RasterSource, ActivateMixin):
    def __init__(self, channel_order, num_channels, raster_transformers=[]):
        super().__init__(channel_order, num_channels, raster_transformers)
        self.mock = Mock()
        self.set_return_vals()

    def set_return_vals(self, raster=None):
        self.mock.get_extent.return_value = Box.make_square(0, 0, 2)
        self.mock.get_dtype.return_value = np.uint8
        self.mock.get_crs_transformer.return_value = IdentityCRSTransformer()
        self.mock._get_chip.return_value = np.random.rand(1, 2, 2, 3)

        if raster is not None:
            self.mock.get_extent.return_value = Box(0, 0, raster.shape[0],
                                                    raster.shape[1])
            self.mock.get_dtype.return_value = raster.dtype

            def get_chip(window):
                return raster[window.ymin:window.ymax, window.xmin:window.
                              xmax, :]

            self.mock._get_chip.side_effect = get_chip

    def get_extent(self):
        return self.mock.get_extent()

    def get_dtype(self):
        return self.mock.get_dtype()

    def get_crs_transformer(self):
        return self.mock.get_crs_transformer()

    def _get_chip(self, window):
        return self.mock._get_chip(window)

    def set_raster(self, raster):
        self.set_return_vals(raster=raster)

    def _activate(self):
        pass

    def _deactivate(self):
        pass


class MockRasterSourceConfig(SupressDeepCopyMixin, RasterSourceConfig):
    def __init__(self, transformers=None, channel_order=None):
        super().__init__(MOCK_SOURCE, transformers, channel_order)
        self.mock = Mock()
        self.mock.to_proto.return_value = None
        self.mock.create_source.return_value = None
        self.mock.update_for_command.return_value = None
        self.mock.save_bundle_files.return_value = (self, [])
        self.mock.load_bundle_files.return_value = self
        self.mock.for_prediction.return_value = self
        self.mock.create_local.return_value = self

    def to_proto(self):
        result = self.mock.to_proto()
        if result is None:
            msg = super().to_proto()
            msg.MergeFrom(RasterSourceConfigMsg(custom_config={}))
            return msg
        else:
            return result

    def create_source(self, tmp_dir):
        result = self.mock.create_source(tmp_dir)
        if result is None:
            self.mock.create_source(tmp_dir)
            transformers = self.create_transformers()
            return MockRasterSource(self.channel_order, 3, transformers)
        else:
            return result

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        super().update_for_command(command_type, experiment_config, context)
        self.mock.update_for_command(command_type, experiment_config, context)

    def report_io(self, command_type, io_def):
        self.mock.report_io(command_type, io_def)

    def save_bundle_files(self, bundle_dir):
        return self.mock.save_bundle_files(bundle_dir)

    def load_bundle_files(self, bundle_dir):
        return self.mock.load_bundle_files(bundle_dir)

    def for_prediction(self, image_uri):
        return self.mock.for_prediction(image_uri)

    def create_local(self, tmp_dir):
        return self.mock.create_local(tmp_dir)


class MockRasterSourceConfigBuilder(SupressDeepCopyMixin,
                                    RasterSourceConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(MockRasterSourceConfig, {})
        self.mock = Mock()
        self.mock.from_proto.return_value = None

    def from_proto(self, msg):
        result = self.mock.from_proto(msg)
        if result is None:
            return super().from_proto(msg)
        else:
            return result


if __name__ == '__main__':
    unittest.main()
