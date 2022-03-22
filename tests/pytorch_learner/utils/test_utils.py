from typing import Callable
import unittest

import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import boto3
from moto import mock_s3

from rastervision.pytorch_learner.utils import (
    compute_conf_mat, compute_conf_mat_metrics, MinMaxNormalize,
    adjust_conv_channels, Parallel, SplitTensor, AddTensors,
    validate_albumentation_transform, A, color_to_triple,
    channel_groups_to_imgs, plot_channel_groups,
    serialize_albumentation_transform, deserialize_albumentation_transform)
from tests.data_files.lambda_transforms import lambda_transforms
from tests import data_file_path


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


class TestMinMaxNormalize(unittest.TestCase):
    def test_tiny_floats(self):
        img = np.random.uniform(0.01, 0.02, (5, 5, 3))
        transform = MinMaxNormalize()
        out = transform(image=img)['image']
        for i in range(3):
            self.assertAlmostEqual(out[:, :, i].min(), 0.0, 6)
            self.assertAlmostEqual(out[:, :, i].max(), 1.0, 6)
            self.assertEqual(out.dtype, np.float32)

    def test_tiny_floats_two_dims(self):
        img = np.random.uniform(0.01, 0.02, (5, 5))
        transform = MinMaxNormalize()
        out = transform(image=img)['image']
        self.assertAlmostEqual(out.min(), 0.0, 6)
        self.assertAlmostEqual(out.max(), 1.0, 6)
        self.assertEqual(out.dtype, np.float32)

    def test_tiny_ints(self):
        img = np.random.uniform(1, 10, (5, 5, 3)).round().astype(np.int32)
        transform = MinMaxNormalize()
        out = transform(image=img)['image']
        for i in range(3):
            self.assertAlmostEqual(out[:, :, i].min(), 0.0, 6)
            self.assertAlmostEqual(out[:, :, i].max(), 1.0, 6)
            self.assertEqual(out.dtype, np.float32)


class TestCustomModules(unittest.TestCase):
    @torch.inference_mode()
    def test_parallel(self):
        m = Parallel(*[nn.Linear(1, 1) for _ in range(5)])
        out = m([torch.randn(10, 1) for _ in range(5)])
        self.assertEqual(len(out), 5)
        for o in out:
            self.assertEqual(o.shape, (10, 1))

    @torch.inference_mode()
    def test_split_tensor(self):
        t = torch.randn(1, 2, 3, 6, 10)

        m = SplitTensor((1, 2, 3), 3)
        ts = m(t)
        self.assertTrue(ts[0].shape, (1, 2, 3, 1, 10))
        self.assertTrue(ts[1].shape, (1, 2, 3, 2, 10))
        self.assertTrue(ts[2].shape, (1, 2, 3, 3, 10))
        self.assertTrue(torch.equal(torch.cat(ts, dim=3), t))

        m = SplitTensor(2, 3)
        ts = m(t)
        self.assertTrue(ts[0].shape, (1, 2, 3, 2, 10))
        self.assertTrue(ts[1].shape, (1, 2, 3, 2, 10))
        self.assertTrue(ts[2].shape, (1, 2, 3, 2, 10))
        self.assertTrue(torch.equal(torch.cat(ts, dim=3), t))

    @torch.inference_mode()
    def test_add_tensors(self):
        m = AddTensors()
        ts = [torch.randn(1, 3, 100, 100) for _ in range(5)]
        self.assertTrue(torch.equal(m(ts), sum(ts)))


class TestAdjustConvChannels(unittest.TestCase):
    def _test_attribs_equal(self, old_conv: nn.Conv2d, new_conv: nn.Conv2d):
        attribs = [
            'out_channels', 'kernel_size', 'stride', 'padding', 'dilation',
            'groups', 'padding_mode'
        ]
        for a in attribs:
            self.assertEqual(getattr(new_conv, a), getattr(old_conv, a))
        self.assertEqual(new_conv.bias is None, old_conv.bias is None)

    def test_noop(self):
        old_conv = nn.Conv2d(3, 64, 5)
        for pretrained in [False, True]:
            new_conv = adjust_conv_channels(old_conv, 3, pretrained=pretrained)
            self.assertEqual(new_conv, old_conv)

    def test_channel_reduction(self):
        old_conv = nn.Conv2d(3, 64, 5)
        # test pretrained=False
        new_conv = adjust_conv_channels(old_conv, 1, pretrained=False)
        self.assertEqual(new_conv.in_channels, 1)
        self._test_attribs_equal(old_conv, new_conv)
        # test pretrained=True
        new_conv = adjust_conv_channels(old_conv, 1, pretrained=True)
        self.assertEqual(new_conv.in_channels, 1)
        self.assertTrue(
            torch.equal(new_conv.weight.data, old_conv.weight.data[:, :1]))
        self._test_attribs_equal(old_conv, new_conv)

    @torch.inference_mode()
    def test_channel_expansion(self):
        old_conv = nn.Conv2d(3, 64, 5)
        # test pretrained=False
        new_conv_1 = adjust_conv_channels(old_conv, 8, pretrained=False)
        self.assertEqual(new_conv_1.in_channels, 8)
        self._test_attribs_equal(old_conv, new_conv_1)
        # test pretrained=True
        new_conv_2 = adjust_conv_channels(old_conv, 8, pretrained=True)
        self.assertIsInstance(new_conv_2, nn.Sequential)
        self.assertIsInstance(new_conv_2[0], SplitTensor)
        self.assertIsInstance(new_conv_2[1], Parallel)
        self.assertIsInstance(new_conv_2[1][0], nn.Conv2d)
        self.assertIsInstance(new_conv_2[1][1], nn.Conv2d)
        self.assertIsInstance(new_conv_2[2], AddTensors)
        self.assertEqual(new_conv_2[1][0], old_conv)
        self.assertEqual(new_conv_2[1][1].in_channels, 5)
        out_1 = new_conv_1(torch.empty(1, 8, 100, 100))
        out_2 = new_conv_2(torch.empty(1, 8, 100, 100))
        self.assertEqual(out_1.shape, out_2.shape)
        self._test_attribs_equal(old_conv, new_conv_2[1][1])


class TestOtherUtils(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_color_to_triple(self):
        rgb = color_to_triple()
        self.assertEqual(len(rgb), 3)
        self.assertTrue(all(0 <= c < 256 for c in rgb))

        rgb = color_to_triple('red')
        self.assertEqual(len(rgb), 3)
        self.assertEqual(rgb, (255, 0, 0))

        self.assertRaises(ValueError, lambda: color_to_triple('not_a_color'))

    def test_albu_serialization_and_deserialization_no_lambda(self):
        x = np.random.randn(20, 20, 3)
        tf_original = A.Resize(10, 10)

        # test if serizlization passes validation check
        tf_serialized = serialize_albumentation_transform(tf_original)
        self.assertEqual(
            validate_albumentation_transform(tf_serialized), tf_serialized)

        # test if the de-serizlized transform's output matches that of the
        # original transform
        tf_deserialized = deserialize_albumentation_transform(tf_serialized)
        np.testing.assert_array_equal(
            tf_original(image=x)['image'], tf_deserialized(image=x)['image'])

    @mock_s3
    def test_albu_serialization_and_deserialization_lambda(self):
        x = np.random.randn(20, 20, 4)
        tf_original = lambda_transforms['ndvi']

        # mock s3 bucket to upload the lambda transforms definition file to
        s3 = boto3.client('s3')
        bucket_name = 'mock_bucket'
        s3.create_bucket(Bucket=bucket_name)
        s3_dir = f's3://{bucket_name}/'

        # test if serizlization passes validation check
        tf_serialized = serialize_albumentation_transform(
            tf_original,
            lambda_transforms_path=data_file_path('lambda_transforms.py'),
            dst_dir=s3_dir)
        self.assertEqual(
            validate_albumentation_transform(tf_serialized), tf_serialized)

        # test if the de-serizlized transform's output matches that of the
        # original transform
        tf_deserialized = deserialize_albumentation_transform(tf_serialized)
        np.testing.assert_array_equal(
            tf_original(image=x)['image'], tf_deserialized(image=x)['image'])

    def test_channel_groups_to_imgs(self):
        imgs = channel_groups_to_imgs(
            torch.rand((100, 100, 3)), {'RGB': (0, 1, 2)})
        self.assertEqual(len(imgs), 1)
        self.assertEqual(imgs[0].shape, (100, 100, 3))

        imgs = channel_groups_to_imgs(
            torch.rand((100, 100, 6)), {
                'RGB': (0, 1, 2),
                'HSV': (3, 4, 5),
                'RBV': (0, 2, 5)
            })
        self.assertEqual(len(imgs), 3)
        self.assertTrue(all(img.shape == (100, 100, 3) for img in imgs))

    def test_plot_channel_groups(self):
        channel_groups = {'RGB': (0, 1, 2)}
        imgs = channel_groups_to_imgs(
            torch.rand((100, 100, 3)), channel_groups)
        _, axs = plt.subplots(1, 1, squeeze=False)
        self.assertNoError(
            lambda: plot_channel_groups(axs[0], imgs, channel_groups))
        plt.close('all')

        channel_groups = {'RGB': (0, 1, 2), 'HSV': (3, 4, 5), 'RBV': (0, 2, 5)}
        imgs = channel_groups_to_imgs(
            torch.rand((100, 100, 6)), channel_groups)
        _, axs = plt.subplots(1, 3, squeeze=False)
        self.assertNoError(
            lambda: plot_channel_groups(axs[0], imgs, channel_groups))
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
