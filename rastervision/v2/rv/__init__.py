# flake8: noqa

from rastervision.v2.rv.box import *
from rastervision.v2.rv.training_data import *


def register_plugin(registry):
    import rastervision.v2.rv.backend
    import rastervision.v2.rv.data
    import rastervision.v2.rv.task
