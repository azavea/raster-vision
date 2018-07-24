import numpy as np

from rastervision.core.label_store import LabelStore
from rastervision.builders import raster_source_builder


class SegmentationRasterFile(LabelStore):
    def __init__(self, src, dst, src_classes=[], dst_classes=[]):
        if src is not None:
            self.src = raster_source_builder.build(src)
        else:
            self.src = None

        correspondence = dict(zip(src_classes, dst_classes))

        def fn(n):
            if n in correspondence:
                return correspondence.get(n)
            else:
                return n

        self.fn = np.vectorize(fn)
        self.correspondence = correspondence

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
