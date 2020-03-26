# flake8: noqa

import rastervision2.pipeline
from rastervision2.core.box import *
from rastervision2.core.data_sample import *
from rastervision2.core.predictor import *
from rastervision2.core.raster_stats import *

# We just need to import anything that contains a Config, so that all
# the register_config decorators will be called which add Configs to the
# registry.
import rastervision2.core.backend
import rastervision2.core.data
import rastervision2.core.rv_pipeline
import rastervision2.core.evaluation
import rastervision2.core.cli


def register_plugin(registry):
    pass
