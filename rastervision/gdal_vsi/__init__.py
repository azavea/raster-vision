# flake8: noqa

import rastervision.pipeline
from rastervision.gdal_vsi.vsi_file_system import VsiFileSystem


def register_plugin(registry):
    registry.set_plugin_version('rastervision.gdal_vsi', 0)
    registry.set_plugin_aliases('rastervision.gdal_vsi',
                                ['rastervision2.gdal_vsi'])
    registry.add_file_system(VsiFileSystem)
