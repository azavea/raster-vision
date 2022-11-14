# flake8: noqa


def register_plugin(registry):
    from rastervision.gdal_vsi.vsi_file_system import VsiFileSystem
    registry.set_plugin_version('rastervision.gdal_vsi', 0)
    registry.add_file_system(VsiFileSystem)


import rastervision.pipeline
from rastervision.gdal_vsi.vsi_file_system import VsiFileSystem

__all__ = [
    VsiFileSystem.__name__,
]
