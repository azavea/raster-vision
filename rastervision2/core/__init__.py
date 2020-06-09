# flake8: noqa


def register_plugin(registry):
    registry.set_plugin_version('rastervision2.core', 0)

    # We just need to import anything that contains a Config, so that all
    # the register_config decorators will be called which add Configs to the
    # registry.
    import rastervision2.core.backend
    import rastervision2.core.data
    import rastervision2.core.rv_pipeline
    import rastervision2.core.evaluation


import rastervision2.pipeline
from rastervision2.core.box import *
from rastervision2.core.data_sample import *
from rastervision2.core.predictor import *
from rastervision2.core.raster_stats import *
