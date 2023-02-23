import unittest

import numpy as np
import torch
from torch import nn
from torchvision.ops import box_convert

from rastervision.pytorch_learner.object_detection_utils import (
    BoxList, collate_fn, TorchVisionODAdapter)


class MockModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x, y=None):
        if self.training:
            assert y is not None
            return {'loss1': 0, 'loss2': 0}
        else:
            N = len(x)
            nboxes = np.random.randint(0, 10)
            outs = [{
                'boxes': torch.rand((nboxes, 4)),
                'labels': torch.randint(0, self.num_classes, (nboxes, )),
                'scores': torch.rand((nboxes, )),
            } for _ in range(N)]
            return outs


class TestTorchVisionODAdapter(unittest.TestCase):
    def test_train_output(self):
        true_num_classes = 3
        model = TorchVisionODAdapter(
            MockModel(num_classes=true_num_classes + 2),
            ignored_output_inds=[0, true_num_classes + 1])
        model.train()

        N = 10
        x = torch.empty(N, 3, 100, 100)
        y = [
            BoxList(
                boxes=torch.rand((10, 4)),
                class_ids=torch.randint(0, 5, (N, ))) for _ in range(N)
        ]
        self.assertRaises(Exception, lambda: model(x))
        out = model(x, y)
        self.assertIsInstance(out, dict)

    def test_eval_output_with_bogus_class(self):
        true_num_classes = 3
        model = TorchVisionODAdapter(
            MockModel(num_classes=true_num_classes + 2),
            ignored_output_inds=[0, true_num_classes + 1])
        model.eval()

        N = 10
        x = torch.empty(N, 3, 100, 100)
        outs = model(x)
        for out in outs:
            self.assertIsInstance(out, BoxList)
            self.assertIn('class_ids', out)
            self.assertIn('scores', out)
            self.assertTrue(
                all((0 <= c < true_num_classes)
                    for c in out.get_field('class_ids')))

    def test_eval_output_without_bogus_class(self):
        true_num_classes = 3
        model = TorchVisionODAdapter(
            MockModel(num_classes=true_num_classes + 1),
            ignored_output_inds=[0])
        model.eval()

        N = 10
        x = torch.empty(N, 3, 100, 100)
        outs = model(x)
        for out in outs:
            self.assertIsInstance(out, BoxList)
            self.assertIn('class_ids', out)
            self.assertIn('scores', out)
            self.assertTrue(
                all((0 <= c < true_num_classes)
                    for c in out.get_field('class_ids')))


class TestBoxList(unittest.TestCase):
    def test_init(self):
        boxes = torch.rand((10, 4))
        # no conversion when format = xyxy
        boxlist = BoxList(boxes)
        self.assertTrue(torch.equal(boxlist.boxes, boxes))
        boxlist = BoxList(boxes, format='xyxy')
        self.assertTrue(torch.equal(boxlist.boxes, boxes))
        # test correct conversion from yxyx to xyxy
        boxlist = BoxList(boxes, format='yxyx')
        self.assertTrue(torch.equal(boxlist.boxes, boxes[:, [1, 0, 3, 2]]))
        # test correct conversion from other formats to xyxy
        for in_fmt in ['xywh', 'cxcywh']:
            boxlist = BoxList(boxes, format=in_fmt)
            self.assertTrue(
                torch.equal(boxlist.boxes, box_convert(boxes, in_fmt, 'xyxy')))

    def test_get_field(self):
        boxes = torch.rand((10, 4))
        class_ids = torch.randint(0, 5, (10, ))
        boxlist = BoxList(boxes, class_ids=class_ids)
        self.assertTrue(torch.equal(boxlist.get_field('class_ids'), class_ids))

    def test_map_extras(self):
        boxes = torch.rand((10, 4))
        class_ids = torch.randint(0, 3, (10, ))
        scores = torch.rand((10, ))
        class_names = np.array(['a', 'b', 'c'])[class_ids.numpy()]
        boxlist = BoxList(
            boxes, class_ids=class_ids, scores=scores, class_names=class_names)
        boxlist = BoxList(
            boxes,
            **boxlist._map_extras(
                func=lambda k, v: v[:-1],
                cond=lambda k, v: torch.is_tensor(v)))
        self.assertTrue(
            torch.equal(boxlist.get_field('class_ids'), class_ids[:-1]))
        self.assertTrue(torch.equal(boxlist.get_field('scores'), scores[:-1]))
        self.assertTrue(all(class_names == boxlist.get_field('class_names')))

    def test_to(self):
        boxes = torch.rand((10, 4))
        class_ids = torch.randint(0, 3, (10, ))
        scores = torch.rand((10, ))
        class_names = np.array(['a', 'b', 'c'])[class_ids.numpy()]
        boxlist = BoxList(
            boxes, class_ids=class_ids, scores=scores, class_names=class_names)
        boxlist = boxlist.to(dtype=torch.float32)
        self.assertTrue(
            torch.equal(boxlist.get_field('class_ids'), class_ids.float()))
        self.assertTrue(
            torch.equal(boxlist.get_field('scores'), scores.float()))
        self.assertTrue(all(class_names == boxlist.get_field('class_names')))

    def test_collate_fn(self):
        imgs = [torch.empty(3, 100, 100) for _ in range(4)]
        boxlists = []
        for _ in range(4):
            boxes = torch.rand((10, 4))
            class_ids = torch.randint(0, 3, (10, ))
            boxlist = BoxList(boxes, class_ids=class_ids)
            boxlists.append(boxlist)
        x, y = collate_fn(zip(imgs, boxes))

        self.assertEqual(x.shape, (4, 3, 100, 100))
        self.assertTrue(all(b1 == b2 for b1, b2 in zip(boxlists, y)))

    def test_scale(self):
        boxes = torch.tensor([
            [0, 0, 1, 1],
            [0, 10, 10, 20],
        ])
        dtype = boxes.dtype

        boxlist = BoxList(boxes.clone())
        yscale, xscale = 1, 1
        boxlist.scale(yscale, xscale)
        torch.testing.assert_close(boxlist.boxes, boxes)

        boxlist = BoxList(boxes.clone())
        yscale, xscale = 5, 3
        boxlist.scale(yscale, xscale)
        x_expected = (boxes[:, [0, 2]] * xscale).to(dtype=dtype)
        y_expected = (boxes[:, [1, 3]] * yscale).to(dtype=dtype)
        torch.testing.assert_close(boxlist.boxes[:, [0, 2]], x_expected)
        torch.testing.assert_close(boxlist.boxes[:, [1, 3]], y_expected)

        boxlist = BoxList(boxes.clone())
        yscale, xscale = 0.5, 0.3
        boxlist.scale(yscale, xscale)
        x_expected = (boxes[:, [0, 2]] * xscale).to(dtype=dtype)
        y_expected = (boxes[:, [1, 3]] * yscale).to(dtype=dtype)
        torch.testing.assert_close(boxlist.boxes[:, [0, 2]], x_expected)
        torch.testing.assert_close(boxlist.boxes[:, [1, 3]], y_expected)


if __name__ == '__main__':
    unittest.main()
