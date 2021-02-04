import unittest

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset

from rastervision.pytorch_learner.utils import (
    compute_conf_mat, compute_conf_mat_metrics, get_base_datasets)


class TestComputeConfMat(unittest.TestCase):
    def test1(self):
        y = torch.tensor([0, 1, 0, 1])
        out = torch.tensor([0, 1, 0, 1])
        num_labels = 2
        conf_mat = compute_conf_mat(out, y, num_labels)
        exp_conf_mat = torch.tensor([[2., 0], [0, 2]])
        self.assertTrue(conf_mat.equal(exp_conf_mat))

    def test2(self):
        y = torch.tensor([0, 1, 0, 1])
        out = torch.tensor([1, 1, 1, 1])
        num_labels = 2
        conf_mat = compute_conf_mat(out, y, num_labels)
        exp_conf_mat = torch.tensor([[0., 2], [0, 2]])
        self.assertTrue(conf_mat.equal(exp_conf_mat))


class TestComputeConfMatMetrics(unittest.TestCase):
    def test1(self):
        label_names = ['a', 'b']
        conf_mat = torch.tensor([[2., 0], [0, 2]])
        metrics = compute_conf_mat_metrics(conf_mat, label_names)
        exp_metrics = {
            'avg_precision': 1.0,
            'avg_recall': 1.0,
            'avg_f1': 1.0,
            'a_precision': 1.0,
            'a_recall': 1.0,
            'a_f1': 1.0,
            'b_precision': 1.0,
            'b_recall': 1.0,
            'b_f1': 1.0
        }
        self.assertDictEqual(metrics, exp_metrics)

    def test2(self):
        label_names = ['a', 'b']
        conf_mat = torch.tensor([[0, 2.], [2, 0]])
        metrics = compute_conf_mat_metrics(conf_mat, label_names)
        exp_metrics = {
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'avg_f1': 0.0,
            'a_precision': 0.0,
            'a_recall': 0.0,
            'a_f1': 0.0,
            'b_precision': 0.0,
            'b_recall': 0.0,
            'b_f1': 0.0
        }
        self.assertDictEqual(metrics, exp_metrics)

    def test3(self):
        label_names = ['a', 'b']
        conf_mat = torch.tensor([[1, 2], [1, 2.]])
        metrics = compute_conf_mat_metrics(conf_mat, label_names, eps=0.0)

        def f1(prec, rec):
            return 2 * (prec * rec) / (prec + rec)

        def mean(a, b):
            return (a + b) / 2

        def round_dict(d):
            return dict([(k, round(v, 3)) for k, v in d.items()])

        a_prec = 1 / 2
        a_rec = 1 / 3
        a_f1 = f1(a_prec, a_rec)
        b_prec = 2 / 4
        b_rec = 2 / 3
        b_f1 = f1(b_prec, b_rec)
        avg_prec = mean(a_prec, b_prec)
        avg_rec = mean(a_rec, b_rec)
        avg_f1 = f1(avg_prec, avg_rec)

        exp_metrics = {
            'avg_precision': avg_prec,
            'avg_recall': avg_rec,
            'avg_f1': avg_f1,
            'a_precision': a_prec,
            'a_recall': a_rec,
            'a_f1': a_f1,
            'b_precision': b_prec,
            'b_recall': b_rec,
            'b_f1': b_f1
        }
        self.assertDictEqual(round_dict(metrics), round_dict(exp_metrics))


class TestOther(unittest.TestCase):
    def test_get_base_datasets(self):
        ds0 = Dataset()
        ds1 = Subset(ds0, [0])
        ds2 = ConcatDataset([ds1, ds1])
        ds3 = Subset(ds2, [0])
        self.assertEqual(get_base_datasets(ds0)[0], ds0)
        self.assertEqual(get_base_datasets(ds1)[0], ds0)
        self.assertEqual(get_base_datasets(ds2)[0], ds0)
        self.assertEqual(get_base_datasets(ds3)[0], ds0)


if __name__ == '__main__':
    unittest.main()
