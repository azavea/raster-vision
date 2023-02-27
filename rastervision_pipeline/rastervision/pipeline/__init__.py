# flake8: noqa
import logging
import json

# This is in a try block since the pytorch_learner plugin (which installs these
# dependencies) may not be installed.
try:
    # torch is imported before anything else in RV in order to avoid a
    # segmentation fault when calling model.to('cuda'). This was determined empirically,
    # and it's not known why this works.
    # sklearn and cv2 are imported here in order to avoid errors that
    # only come up when importing rastervision inside of a Jupyter notebook cell.
    # Doing these imports first only appears to fixes the problem if it's in this
    # package, and not the `pytorch_learner` or `pytorch_core`.
    # Without sklearn:
    # ImportError: /opt/conda/lib/python3.9/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
    # Without cv2:
    # ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block
    # See https://github.com/opencv/opencv/issues/14884
    import torch
    import sklearn
    import cv2
except:  # pragma: no cover
    pass

from rastervision.pipeline.rv_config import RVConfig
from rastervision.pipeline.registry import Registry
from rastervision.pipeline.verbosity import Verbosity

root_logger = logging.getLogger('rastervision')
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s: %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

rv_config_ = RVConfig()
registry_ = Registry()
registry_.load_plugins()
registry_.load_builtins()
