import unittest

from rastervision.backend.torch_utils.metrics import (compute_conf_mat,
                                                      compute_conf_mat_metrics)
import rastervision as rv


@unittest.skipIf(not rv.backend.pytorch_available, 'PyTorch is not available')
class TestComputeConfMat(unittest.TestCase):
    def test1(self):
        import torch
        y = torch.tensor([0, 1, 0, 1])
        out = torch.tensor([0, 1, 0, 1])
        num_labels = 2
        conf_mat = compute_conf_mat(out, y, num_labels)
        exp_conf_mat = torch.tensor([[2., 0], [0, 2]])
        self.assertTrue(conf_mat.equal(exp_conf_mat))

    def test2(self):
        import torch
        y = torch.tensor([0, 1, 0, 1])
        out = torch.tensor([1, 1, 1, 1])
        num_labels = 2
        conf_mat = compute_conf_mat(out, y, num_labels)
        exp_conf_mat = torch.tensor([[0., 2], [0, 2]])
        self.assertTrue(conf_mat.equal(exp_conf_mat))


@unittest.skipIf(not rv.backend.pytorch_available, 'PyTorch is not available')
class TestComputeConfMatMetrics(unittest.TestCase):
    def test1(self):
        import torch
        conf_mat = torch.tensor([[2., 0], [0, 2]])
        metrics = compute_conf_mat_metrics(conf_mat)
        exp_metrics = {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        self.assertDictEqual(metrics, exp_metrics)

    def test2(self):
        import torch
        conf_mat = torch.tensor([[0, 2.], [2, 0]])
        metrics = compute_conf_mat_metrics(conf_mat)
        exp_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        self.assertDictEqual(metrics, exp_metrics)

    def test3(self):
        import torch
        conf_mat = torch.tensor([[1, 2], [1, 2.]])
        metrics = compute_conf_mat_metrics(conf_mat)
        exp_metrics = {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
        self.assertDictEqual(metrics, exp_metrics)


if __name__ == '__main__':
    unittest.main()
