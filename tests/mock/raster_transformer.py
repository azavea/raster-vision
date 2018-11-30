import unittest
from unittest.mock import Mock

import rastervision as rv
from rastervision.data import (
    RasterTransformerConfig, RasterTransformerConfigBuilder, RasterTransformer)
from rastervision.protos.raster_transformer_pb2 \
    import RasterTransformerConfig as RasterTransformerConfigMsg

from tests.mock import SupressDeepCopyMixin

MOCK_TRANSFORMER = 'MOCK_TRANSFORMER'


class MockRasterTransformer(RasterTransformer):
    def __init__(self):
        self.mock = Mock()

        self.mock.transform.return_value = None

    def transform(self, chip, channel_order=None):
        result = self.mock.transform(chip, channel_order)
        if result is None:
            return chip
        else:
            return result


class MockRasterTransformerConfig(SupressDeepCopyMixin,
                                  RasterTransformerConfig):
    def __init__(self):
        super().__init__(MOCK_TRANSFORMER)
        self.mock = Mock()

        self.mock.to_proto.return_value = None
        self.mock.create_transformer.return_value = None
        self.mock.update_for_command.return_value = None
        self.mock.save_bundle_files.return_value = (self, [])
        self.mock.load_bundle_files.return_value = self
        self.mock.for_prediction.return_value = self
        self.mock.create_local.return_value = self

    def to_proto(self):
        result = self.mock.to_proto()
        if result is None:
            msg = RasterTransformerConfigMsg(
                transformer_type=self.transformer_type, custom_config={})
            return msg
        else:
            return result

    def create_transformer(self):
        result = self.mock.create_transformer()
        if result is None:
            return MockRasterTransformer()
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

    def for_prediction(self, image_uri):
        return self.mock.for_prediction(image_uri)

    def create_local(self, tmp_dir):
        return self.mock.create_local(tmp_dir)


class MockRasterTransformerConfigBuilder(SupressDeepCopyMixin,
                                         RasterTransformerConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(MockRasterTransformerConfig, {})
        self.mock = Mock()

        self.mock.from_proto.return_value = None

    def from_proto(self, msg):
        result = self.mock.from_proto(msg)
        if result is None:
            retval = super().from_proto(msg)
            if not retval and msg.transformer_type == 'MOCK_TRANSFORMER':
                retval = MockRasterTransformerConfigBuilder()
            return retval
        else:
            return result


if __name__ == '__main__':
    unittest.main()
