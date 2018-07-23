from rastervision.core.label_store import LabelStore
from rastervision.raster_sources.geotiff_files import GeoTiffFiles
from rastervision.builders import raster_source_builder


class SegmentationRasterFile(LabelStore):

    def __init__(self, src, dst, src_classes=[], dst_classes=[]):
        if src is not None:
            self.src = raster_source_builder.build(src)
        else:
            self.src = None
        pass

    def clear(self):
        pass

    def set_labels(self, labels):
        pass

    def get_labels(self, window=None):
        pass

    def extend(self, labels):
        pass

    def save(self):
        pass
