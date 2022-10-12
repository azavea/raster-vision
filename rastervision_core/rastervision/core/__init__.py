# flake8: noqa

# torch needs to be imported before anything else in RV or we will get a
# segmentation fault when calling model.to('cuda'). This is very weird, and not
# a great solution, but the show must go on.
try:
    # This is in a try block in case RV is being used without the torch backend
    # plugin and torch is not installed
    import torch
except:
    pass


def register_plugin(registry):
    registry.set_plugin_version('rastervision.core', 7)
    registry.set_plugin_aliases('rastervision.core', ['rastervision2.core'])
    from rastervision.core.cli import predict
    registry.add_plugin_command(predict)


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
