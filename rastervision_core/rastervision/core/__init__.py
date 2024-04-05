# flake8: noqa


def register_plugin(registry):
    registry.set_plugin_version('rastervision.core', 13)
    from rastervision.core.cli import predict, predict_scene
    registry.add_plugin_command(predict)
    registry.add_plugin_command(predict_scene)


import rastervision.pipeline
from rastervision.core.box import *
from rastervision.core.data_sample import *
from rastervision.core.predictor import *
from rastervision.core.raster_stats import *

# We just need to import anything that contains a Config, so that all
# the register_config decorators will be called which add Configs to the
# registry.
import rastervision.core.backend
import rastervision.core.data
import rastervision.core.rv_pipeline
import rastervision.core.evaluation
