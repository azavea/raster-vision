import unittest

from rastervision.pytorch_learner import Backbone, ObjectDetectionModelConfig


class TestObjectDetectionModelConfig(unittest.TestCase):
    def test_extra_args(self):
        cfg = ObjectDetectionModelConfig(
            backbone=Backbone.resnet18,
            pretrained=False,
            extra_args=dict(box_nms_thresh=0.4))
        model = cfg.build_default_model(
            num_classes=2, in_channels=3, img_sz=256)
        self.assertEqual(model.roi_heads.nms_thresh, 0.4)
