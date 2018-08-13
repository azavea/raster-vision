import numpy as np

from typing import (Dict, Union)

from rastervision.builders import raster_source_builder
from rastervision.core.box import Box
from rastervision.core.label_store import LabelStore
from rastervision.core.raster_source import RasterSource
from rastervision.protos.raster_source_pb2 import (RasterSource as
                                                   RasterSourceProto)
from rastervision.utils.misc import color_to_integer

RasterUnion = Union[RasterSource, RasterSourceProto, str, None]


class SegmentationRasterFile(LabelStore):
    """A label store for segmentation raster files.

    """

    def __init__(self,
                 src: RasterUnion,
                 dst: RasterUnion,
                 raster_class_map: Dict[str, int] = {}):
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

        if isinstance(raster_class_map, dict):
            src_classes = list(
                map(lambda c: color_to_integer(c), raster_class_map.keys()))
            rv_classes = list(raster_class_map.values())
        else:
            src_classes = list(
                map(lambda c: color_to_integer(c.source_class),
                    raster_class_map))
            rv_classes = list(
                map(lambda c: c.raster_vision_class, raster_class_map))
        src_to_rv_class_map = dict(zip(src_classes, rv_classes))
        self.src_classes = src_classes
        self.rv_classes = rv_classes

        def src_to_rv(n: int) -> int:
            """Translate source classes to raster vision classes.

            args:
                 n: A source class represented as a packed rgb pixel
                      (an integer in the range 0 to 2**24-1).

            Returns:
                 The destination class as an integer.

            """
            if n in src_to_rv_class_map:
                return src_to_rv_class_map.get(n)
            else:
                return 0

        self.src_to_rv = np.vectorize(src_to_rv, otypes=[np.uint8])

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

    def has_labels(self, window: Box) -> bool:
        """Given a window, deterine whether there are any labels in it.

        Args:
             window: A window given as a Box object.

        Returns:
             Either True or False.

        """
        if self.src is not None:
            labels = self.get_labels(window)
            return np.in1d(self.src_classes, labels).any()
        else:
            return False

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
            labels = self.src._get_chip(window)
            r = np.array(labels[:, :, 0], dtype=np.uint32) * (1 << 16)
            g = np.array(labels[:, :, 1], dtype=np.uint32) * (1 << 8)
            b = np.array(labels[:, :, 2], dtype=np.uint32) * (1 << 0)
            packed = r + g + b
            return self.src_to_rv(packed)
        else:
            ymin = window.ymin
            xmin = window.xmin
            ymax = window.ymax
            xmax = window.xmax
            return np.zeros((xmax - xmin, ymax - ymin), dtype=np.uint8)

    def extend(self, labels):
        pass

    def save(self):
        pass
