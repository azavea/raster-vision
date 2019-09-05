import unittest

import rastervision as rv


@unittest.skipIf(not rv.backend.pytorch_available, 'PyTorch is not available')
class TestPyTorchChipClassificationConfig(unittest.TestCase):
    def test_builder(self):
        batch_size = 10
        num_epochs = 10

        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                            .with_chip_size(200) \
                            .with_classes({
                                'building': (1, 'red'),
                                'no_building': (2, 'black')
                            }) \
                            .build()

        backend = rv.BackendConfig.builder(rv.PYTORCH_CHIP_CLASSIFICATION) \
            .with_task(task) \
            .with_train_options(
                batch_size=batch_size,
                num_epochs=num_epochs) \
            .build()

        msg = backend.to_proto()
        backend = rv.BackendConfig.builder(rv.PYTORCH_CHIP_CLASSIFICATION) \
            .from_proto(msg).build()

        self.assertEqual(backend.train_opts.batch_size, batch_size)
        self.assertEqual(backend.train_opts.num_epochs, num_epochs)


if __name__ == '__main__':
    unittest.main()
