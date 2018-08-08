import numpy as np

from typing import (List, Union)
from PIL import ImageColor

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
                 src_classes: List[str] = [],
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

        def color_to_integer(color: str) -> int:
            """Given a PIL ImageColor string, return a packed integer.

            Args:
                 color: A PIL ImageColor string

            Returns:
                 An integer containing the packed RGB values.

            """
            try:
                triple = ImageColor.getrgb(color)
            except ValueError:
                r = np.random.randint(0, 256)
                g = np.random.randint(0, 256)
                b = np.random.randint(0, 256)
                triple = (r, g, b)

            r = triple[0] * (1 << 16)
            g = triple[1] * (1 << 8)
            b = triple[2] * (1 << 0)
            integer = r + g + b
            return integer

        src_classes = list(map(color_to_integer, src_classes))
        correspondence = dict(zip(src_classes, dst_classes))

        def src_to_dst(n: int) -> int:
            """Translate source classes to destination class.

            args:
                 n: A source class represented as a packed rgb pixel
                      (an integer in the range 0 to 2**24-1).

            Returns:
                 The destination class as an integer.

            """
            if n in correspondence:
                return correspondence.get(n)
            else:
                return 0

        self.src_to_dst = np.vectorize(src_to_dst)

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

    def get_labels(self, window: Box) -> np.ndarray:
        """Get labels from a window.

        If self.src is not None then a label window is clipped from
        it.  If self.src is None then return an appropriatly shaped
        np.ndarray of zeros.

        Args:
             window: A window given as a Box object.

        Returns:
             np.ndarray

        """
        if self.src is not None:
            return self.src._get_chip(window)
        else:
            ymin = window.ymin
            xmin = window.xmin
            ymax = window.ymax
            xmax = window.xmax
            return np.zeros((ymax - ymin, xmax - xmin, self.channels))

    def extend(self, labels):
        pass

    def save(self):
        pass
