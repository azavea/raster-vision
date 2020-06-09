import warnings
warnings.filterwarnings('ignore')  # noqa
from os.path import join, isdir, basename
import logging
import glob
import requests
import boto3
from urllib.parse import urlparse
import codecs

import numpy as np
import matplotlib
matplotlib.use('Agg')  # noqa
import torch
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import torchvision

from rastervision2.pytorch_learner.learner import Learner
from rastervision2.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, color_to_triple)

from rastervision2.pytorch_learner.semantic_segmentation_learner import SemanticSegmentationLearner

log = logging.getLogger(__name__)


def make_model(band_count, input_stride=1, class_count=1, divisor=1, pretrained=False):
    raise NotImplementedError()


def read_text(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme.startswith('http'):
        return requests.get(uri).text
    elif parsed.scheme.startswith('s3'):
        parsed2 = urlparse(uri, allow_fragments=False)
        bucket = parsed2.netloc
        prefix = parsed2.path.lstrip('/')
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, prefix)
        return obj.get()['Body'].read().decode('utf-8')
    else:
        with codecs.open(uri, encoding='utf-8', mode='r') as f:
            return f.read()


class ResearchArchitectureLearner(SemanticSegmentationLearner):

    def build_model(self):
        pretrained = self.cfg.pretrained
        uri = self.cfg.architecture
        bands = self.cfg.bands

        arch_str = read_text(uri)
        arch_code = compile(arch_str, uri, 'exec')
        exec(arch_code, globals())

        model = make_model(
            bands,
            input_stride=1,
            class_count=len(self.cfg.data.class_names),
            divisor=1,
            pretrained=pretrained
        )
        return model

    def post_forward(self, pred):
        pred_seg = pred.get('seg', pred.get('out', None))
        return pred_seg
