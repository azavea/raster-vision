import numpy as np

from rastervision.core.box import Box
from rastervision.core.label_store import LabelStore
from rastervision.core.raster_source import RasterSource
from rastervision.builders import raster_source_builder
from rastervision.protos.raster_source_pb2 import (RasterSource as
                                                   RasterSourceProto)


class SegmentationRasterFile(LabelStore):
    def __init__(self, src, dst, src_classes=[], dst_classes=[]):
        self.set_labels(src)

        if isinstance(dst, RasterSource):
            self.dst = dst
        elif isinstance(dst, RasterSourceProto):
            self.dst = raster_source_builder.build(dst)
        elif dst is None or dst is '':
            self.dst = None
        elif isinstance(dst, str):
            pass  # XXX str instead of RasterSourceProto
        else:
            raise ValueError('Unsure how to handle dst={}'.format(type(dst)))

        correspondence = dict(zip(src_classes, dst_classes))

        def fn(n):
            if n in correspondence:
                return correspondence.get(n)
            else:
                return n

        self.fn = np.vectorize(fn)
        self.correspondence = correspondence

    def clear(self):
        self.src = None

    def set_labels(self, src):
        if isinstance(src, RasterSource):
            self.src = src
        elif isinstance(src, RasterSourceProto):
            self.src = raster_source_builder.build(src)
        elif src is None:
            self.src = None
        else:
            raise ValueError('Unsure how to handle src={}'.format(type(src)))

        if self.src is not None:
            small_box = Box(0, 0, 1, 1)
            (self.channels, _, _) = self.src._get_chip(small_box).shape
        else:
            self.channels = 1

    def get_labels(self, window=None):
        if self.src is not None:
            return self.src._get_chip(window)
        else:
            ymin = window.ymin
            xmin = window.xmin
            ymax = window.ymax
            xmax = window.xmax
            return np.zeros((self.channels, ymax - ymin, xmax - xmin))

    def extend(self, labels):
        pass

    def save(self):
        pass
