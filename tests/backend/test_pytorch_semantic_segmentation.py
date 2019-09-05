import unittest

import rastervision as rv


@unittest.skipIf(not rv.backend.pytorch_available, 'PyTorch is not available')
class TestPyTorchSemanticSegmentationConfig(unittest.TestCase):
    def test_builder(self):
        batch_size = 10
        num_epochs = 10

        chip_size = 300
        class_map = {'red': (1, 'red'), 'green': (2, 'green')}
        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(chip_size) \
                            .with_classes(class_map) \
                            .with_chip_options(window_method='sliding',
                                               stride=chip_size,
                                               debug_chip_probability=1.0) \
                            .build()

        backend = rv.BackendConfig.builder(rv.PYTORCH_SEMANTIC_SEGMENTATION) \
            .with_task(task) \
            .with_train_options(
                batch_size=batch_size,
                num_epochs=num_epochs) \
            .build()

        msg = backend.to_proto()
        backend = rv.BackendConfig.builder(rv.PYTORCH_SEMANTIC_SEGMENTATION) \
            .from_proto(msg).build()

        self.assertEqual(backend.train_opts.batch_size, batch_size)
        self.assertEqual(backend.train_opts.num_epochs, num_epochs)


if __name__ == '__main__':
    unittest.main()
