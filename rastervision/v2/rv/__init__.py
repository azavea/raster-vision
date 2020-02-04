# flake8: noqa

from rastervision.v2.rv.box import *
from rastervision.v2.rv.data_sample import *
from rastervision.v2.rv.predictor import *


def register_plugin(registry):
    import rastervision.v2.rv.backend
    import rastervision.v2.rv.data
    import rastervision.v2.rv.task
