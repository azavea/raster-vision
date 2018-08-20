import numpy as np

from typing import (Dict, Union)

from rastervision.builders import raster_source_builder
from rastervision.core.box import Box
from rastervision.core.label_store import LabelStore
from rastervision.core.raster_source import RasterSource
from rastervision.protos.raster_source_pb2 import (RasterSource as RasterSourceProto)
from rastervision.utils.files import (make_dir)
from rastervision.utils.misc import color_to_integer
from typing import (Dict, List, Tuple)

RasterUnion = Union[RasterSource, RasterSourceProto, str, None]


class SegmentationRasterFile(LabelStore):
    """A label store for segmentation raster files.

    """

    def __init__(self,
                 src: RasterUnion,
                 dst: Union[str, None],
                 raster_class_map: Dict[str, int] = {}):
        """Constructor.

        Args:
             src: A source of raster label data (either an object that
                  can provide it or a path).
             dst: A destination for raster label data.
             raster_class_map: A mapping between the labels found in
                  the source (the given labels) and those desired in
                  the destination (those produced by the predictions).

        """
        self.set_labels(src)
        self.label_pairs = []
        self.raster_class_map = raster_class_map

        if dst is None or dst is '':
            self.dst = None
        elif isinstance(dst, str):
            self.dst = dst
        else:
            raise ValueError('Unsure how to handle dst={}'.format(type(dst)))

        assert (self.src is None) ^ (self.dst is None)

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

            Args:
                 n: A source class represented as a packed RGB pixel
                      (an integer in the range 0 to 2**24-1).

            Returns:
                 The destination class as an integer.

            """
            if n in src_to_rv_class_map:
                return src_to_rv_class_map.get(n)
            else:
                return 0x00

        self.src_to_rv = np.vectorize(src_to_rv, otypes=[np.uint8])

    def clear(self):
        """Clear all labels."""
        self.src = None

    def set_labels(self, src: Union[RasterUnion, List[Tuple[Box, np.ndarray]]]) -> None:
        """Set labels, overwriting any that existed prior to this call.

        Args:
             src: A source of raster label data (either an object that
                  can provide it, a path, or a list of window × array
                  pairs).

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

        # if self.src is not None:
        #     small_box = Box(0, 0, 1, 1)
        #     (self.channels, _, _) = self.src._get_chip(small_box).shape
        # else:
        #     self.channels = 1
        self.channels = 3  # Only support three-channel images

    def interesting_subwindow(self, window: Box, size: int, shift: int) -> Union[Box, None]:
        """Given a larger window, return a sub-window that contains interesting
        pixels (pixels of class 1).

        Args:
             window: The larger window from-which the sub-window will
                  be clipped.
             size: The linear size (height and width) of the smaller
                  window.
             shift: How far to shift the returned window to the
                  right.  This is useful because the returned window
                  is constructed by finding the first pixel from the
                  top-left, which can result in objects being cutoff.

        Returns:
             Either a sub-window containing interesting pixels or None
             if no such window can be found.

        """
        if self.src is not None:
            larger_size = window.xmax - window.xmin  # XXX assumed square

            labels = self.src._get_chip(window)
            r = np.array(labels[:, :, 0], dtype=np.uint32) * (1 << 16)
            g = np.array(labels[:, :, 1], dtype=np.uint32) * (1 << 8)
            b = np.array(labels[:, :, 2], dtype=np.uint32) * (1 << 0)
            packed = r + g + b
            translated = self.src_to_rv(packed)
            argmax = np.argmax(translated == 1)

            if argmax == 0:
                return None
            else:
                major = int(argmax / larger_size)
                minor = argmax - (major * larger_size)
                old_ymax = window.ymax
                old_xmax = window.xmax
                new_ymin = window.ymin + major
                new_xmin = window.xmin + minor - shift
                ymin = new_ymin if (new_ymin + size <=
                                    old_ymax) else old_ymax - size
                xmin = new_xmin if (new_xmin + size <=
                                    old_xmax) else old_ymax - size
                retval = Box(
                    ymin=ymin, xmin=xmin, ymax=ymin + size, xmax=xmin + size)
                return retval
        else:
            return None

    def has_labels(self, window: Box) -> bool:  # XXX
        """Given a window, determine whether there are any labels in it.

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

    def get_labels(self, window: Union[Box, None] = None) -> Union[np.ndarray, None]:
        """Get labels from a window.

        If self.src is not None then a label window is clipped from
        it.  If self.src is None then return an appropriately shaped
        np.ndarray of zeros.

        Args:
             window: A window given as a Box object or None.

        Returns:
             np.ndarray

        """
        if self.src is not None and window is not None:
            labels = self.src._get_chip(window)
            r = np.array(labels[:, :, 0], dtype=np.uint32) * (1 << 16)
            g = np.array(labels[:, :, 1], dtype=np.uint32) * (1 << 8)
            b = np.array(labels[:, :, 2], dtype=np.uint32) * (1 << 0)
            packed = r + g + b
            return self.src_to_rv(packed)
        elif window is not None:
            ymin = window.ymin
            xmin = window.xmin
            ymax = window.ymax
            xmax = window.xmax
            return np.zeros((xmax - xmin, ymax - ymin), dtype=np.uint8)
        else:
            return None

    def extend(self, labels: Tuple[Box, np.ndarray]) -> None:
        """Add incoming labels to the list of labels.

        Args:
             labels: A Box × np.ndarray pair.

        Returns:
             None.

        """
        self.label_pairs.append(labels)

    def save(self):
        import rasterio

        make_dir(self.dst, use_dirname=True)  # XXX not remote safe

        boxes = list(map(lambda c: c[0], self.label_pairs))
        xmax = max(map(lambda b: b.xmax, boxes))
        ymax = max(map(lambda b: b.ymax, boxes))

        # https://github.com/mapbox/rasterio/blob/master/docs/quickstart.rst
        # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        with rasterio.open(self.dst, 'w', driver='GTiff', height=ymax, width=xmax, count=1, dtype=np.uint8) as dataset:
            for (box, data) in self.label_pairs:
                window = (box.ymin, box.ymax), (box.xmin, box.xmax)
                dataset.write_band(1, np.array(data*42, dtype=np.uint8), window=window)
