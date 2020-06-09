import codecs
import logging
from urllib.parse import urlparse

import boto3
import requests
import torch  # noqa
import torch.nn.functional as F
import torchvision  # noqa
from rastervision2.pytorch_learner.semantic_segmentation_learner import \
    SemanticSegmentationLearner

log = logging.getLogger(__name__)


def make_model(band_count,
               input_stride=1,
               class_count=1,
               divisor=1,
               pretrained=False):
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
        resolution_divisor = self.cfg.resolution_divisor

        arch_str = read_text(uri)
        arch_code = compile(arch_str, uri, 'exec')
        exec(arch_code, globals())

        model = make_model(
            bands,
            input_stride=1,
            class_count=len(self.cfg.data.class_names),
            divisor=resolution_divisor,
            pretrained=pretrained)
        return model

    def train_step(self, batch, batch_ind):
        x, y = batch
        x = self.model(x)
        seg = x.get('seg', x.get('out', None))
        aux = x.get('aux', None)
        if seg is not None and aux is not None:
            return {
                'train_loss':
                F.cross_entropy(seg, y) * .4 * F.cross_entropy(aux, y)
            }
        elif seg is not None and aux is None:
            return {'train_loss': F.cross_entropy(seg, y)}
        else:
            raise NotImplementedError()

    def post_forward(self, x):
        seg = x.get('seg', x.get('out', None))
        return seg

    def prob_to_pred(self, x):
        return x.argmax(1)
