# flake8: noqa

import rastervision2.pipeline
from rastervision2.gdal_vsi.vsi_file_system import VsiFileSystem


def register_plugin(registry):
    registry.add_file_system(VsiFileSystem)
