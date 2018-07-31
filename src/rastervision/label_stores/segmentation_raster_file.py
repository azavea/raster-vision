import numpy as np

from typing import (List, Union)

from rastervision.core.box import Box
from rastervision.core.label_store import LabelStore
from rastervision.core.raster_source import RasterSource
from rastervision.builders import raster_source_builder
from rastervision.protos.raster_source_pb2 import (RasterSource as
                                                   RasterSourceProto)

RasterUnion = Union[RasterSource, RasterSourceProto, str, None]


class SegmentationRasterFile(LabelStore):
    """A label store for segmentation raster files.

    """

    def __init__(self,
                 src: RasterUnion,
                 dst: RasterUnion,
                 src_classes: List[int] = [],
                 dst_classes: List[int] = []):
        """Constructor.

        Args:
             src: A source of raster label data (either an object that
                  can provide it or a path).
             dst: A destination for raster label data.
             src_classes: A list of integer classes found in the label
                  source.  These are zipped with the destination list
                  to produce a correspondence between input classes
                  and output classes.
             dst_classes: A list of integer classes found in the
                  labels that are to be produced.  These labels should
                  match those given in the workflow configuration file
                  (the class map).

        """
        self.set_labels(src)

        if isinstance(dst, RasterSource):
            self.dst = dst
        elif isinstance(dst, RasterSourceProto):
            self.dst = raster_source_builder.build(dst)
        elif dst is None or dst is '':
            self.dst = None
        elif isinstance(dst, str):
            pass  # XXX seeing str instead of RasterSourceProto
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
        """Clear all labels."""
        self.src = None

    def set_labels(self, src: RasterUnion) -> None:
        """Set labels, overwriting any that existed prior to this call.

        Args:
             src: A source of raster label data (either an object that
                  can provide it or a path).

        Returns:
             None

        """
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

    def get_labels(self, window: Union[Box, None] = None) -> np.ndarray:
        """Get labels from a window or from the entire scene.

        Args:
             window: Either a window (given as a Box object) or None.
                  In the former case labels are returned for the
                  windowed area, in the latter case labels are
                  returned for the entire scene.

        Returns:
             numpy.ndarray

        """
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
