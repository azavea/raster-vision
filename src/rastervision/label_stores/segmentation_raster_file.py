import math
import numpy as np
import os
import tempfile

from typing import (Dict, List, Tuple, Union)
from urllib.parse import urlparse

from rastervision.builders import raster_source_builder
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap
from rastervision.core.label_store import LabelStore
from rastervision.core.raster_source import RasterSource
from rastervision.protos.raster_source_pb2 import (RasterSource as
                                                   RasterSourceProto)
from rastervision.utils.files import (get_local_path, make_dir, sync_dir)
from rastervision.utils.misc import (color_to_integer, color_to_triple)

RasterUnion = Union[RasterSource, RasterSourceProto, str, None]


class SegmentationRasterFile(LabelStore):
    """A label store for segmentation raster files.

    """

    def __init__(self,
                 source: RasterUnion,
                 sink: Union[str, None],
                 class_map: Union[ClassMap, None],
                 raster_class_map: Dict[str, int] = {}):
        """Constructor.

        Args:
             source: A source of raster label data (either an object that
                  can provide it or a path).
             sink: A destination for raster label data.
             class_map: A class map object used for producing output
                  rasters.
             raster_class_map: A mapping between the labels found in
                  the source (the given labels) and those desired in
                  the destination (those produced by the predictions).

        """
        self.set_labels(source)
        self.label_pairs = []
        self.raster_class_map = raster_class_map
        self.class_map = class_map

        if sink is None or sink is '':
            self.sink = None
        elif isinstance(sink, str):
            self.sink = sink
        else:
            raise ValueError('Unsure how to handle sink={}'.format(type(sink)))

        assert (self.source is None) ^ (self.sink is None)

        if isinstance(raster_class_map, dict):
            source_classes = list(
                map(lambda c: color_to_integer(c), raster_class_map.keys()))
            rv_classes = list(raster_class_map.values())
        else:
            source_classes = list(
                map(lambda c: color_to_integer(c.source_class),
                    raster_class_map))
            rv_classes = list(
                map(lambda c: c.raster_vision_class, raster_class_map))
        source_to_rv_class_map = dict(zip(source_classes, rv_classes))
        self.source_classes = source_classes
        self.rv_classes = rv_classes

        def source_to_rv(n: int) -> int:
            """Translate source classes to raster vision classes.

            Args:
                 n: A source class represented as a packed RGB pixel
                      (an integer in the range 0 to 2**24-1).

            Returns:
                 The destination class as an integer.

            """
            if n in source_to_rv_class_map:
                return source_to_rv_class_map.get(n)
            else:
                return 0x00

        self.source_to_rv = np.vectorize(source_to_rv, otypes=[np.uint8])

    def clear(self):
        """Clear all labels."""
        self.source = None

    def set_labels(
            self,
            source: Union[RasterUnion, List[Tuple[Box, np.ndarray]]]) -> None:
        """Set labels, overwriting any that existed prior to this call.

        Args:
             source: A source of raster label data (either an object that
                  can provide it, a path, or a list of window × array
                  pairs).

        Returns:
             None

        """
        if isinstance(source, RasterSource):
            self.source = source
        elif isinstance(source, RasterSourceProto):
            self.source = raster_source_builder.build(source)
        elif source is None:
            self.source = None
        else:
            raise ValueError('Unsure how to handle source={}'.format(
                type(source)))

    def interesting_subwindow(self, window: Box, size: int,
                              shift: int) -> Union[Box, None]:
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
        if self.source is not None:
            larger_size = window.xmax - window.xmin  # XXX assumed square

            labels = self.source._get_chip(window)
            r = np.array(labels[:, :, 0], dtype=np.uint32) * (1 << 16)
            g = np.array(labels[:, :, 1], dtype=np.uint32) * (1 << 8)
            b = np.array(labels[:, :, 2], dtype=np.uint32) * (1 << 0)
            packed = r + g + b
            translated = self.source_to_rv(packed)
            argmax = np.argmax(translated == 1)

            if translated.sum() < (larger_size * math.sqrt(larger_size)):
                return -2
            elif argmax == 0 and not translated[0, 0] == 1:
                return -1
            elif size == larger_size:
                return window
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

    def get_labels(self,
                   window: Union[Box, None] = None) -> Union[np.ndarray, None]:
        """Get labels from a window.

        If self.source is not None then a label window is clipped from
        it.  If self.source is None then return an appropriately shaped
        np.ndarray of zeros.

        Args:
             window: A window given as a Box object or None.

        Returns:
             np.ndarray

        """
        if self.source is not None and window is not None:
            labels = self.source._get_chip(window)
            r = np.array(labels[:, :, 0], dtype=np.uint32) * (1 << 16)
            g = np.array(labels[:, :, 1], dtype=np.uint32) * (1 << 8)
            b = np.array(labels[:, :, 2], dtype=np.uint32) * (1 << 0)
            packed = r + g + b
            return self.source_to_rv(packed)
        elif window is not None:
            ymin = window.ymin
            xmin = window.xmin
            ymax = window.ymax
            xmax = window.xmax
            return np.zeros((xmax - xmin, ymax - ymin), dtype=np.uint8)
        else:
            return None

    def extend(self, labels: List[Tuple[Box, np.ndarray]]) -> None:
        """Add incoming labels to the list of labels.

        Args:
             labels: A list of Box × np.ndarray pairs.

        Returns:
             None.

        """
        self.label_pairs.extend(labels)

    def save(self):
        """Save the labels to a GeoTiff raster at the location pointed-to by
        self.sink.

        """
        import rasterio

        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        local_sink = get_local_path(self.sink, temp_dir)

        make_dir(local_sink, use_dirname=True)

        boxes = list(map(lambda c: c[0], self.label_pairs))
        xmax = max(map(lambda b: b.xmax, boxes))
        ymax = max(map(lambda b: b.ymax, boxes))

        rv_classes = self.class_map.get_keys()
        colors = map(lambda c: c.color, self.class_map.get_items())
        triples = list(
            map(lambda c: np.array(color_to_triple(c), dtype=np.uint8),
                colors))
        mapping = dict(zip(rv_classes, triples))

        def class_to_channel(channel: int, c: int) -> int:
            """Given a channel (red, green, or blue) and a class, return the
            intensity of that channel.

            Args:

                 channel: An integer with value 0, 1, or 2
                      representing the channel.
                 c: The class value represented as an integer.

            Returns:
                 The intensity of the channel for the color associated
                      with the given class.

            """
            if c in mapping:
                return mapping.get(c)[channel]
            else:
                return 0x00

        class_to_r = np.vectorize(
            lambda c: class_to_channel(0, c), otypes=[np.uint8])
        class_to_g = np.vectorize(
            lambda c: class_to_channel(1, c), otypes=[np.uint8])
        class_to_b = np.vectorize(
            lambda c: class_to_channel(2, c), otypes=[np.uint8])
        class_to = [class_to_r, class_to_g, class_to_b]

        # https://github.com/mapbox/rasterio/blob/master/docs/quickstart.rst
        # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        with rasterio.open(
                local_sink,
                'w',
                driver='GTiff',
                height=ymax,
                width=xmax,
                count=3,
                dtype=np.uint8) as dataset:
            for (box, data) in self.label_pairs:
                window = (box.ymin, box.ymax), (box.xmin, box.xmax)
                for chan in range(3):
                    pixels = class_to[chan](data)
                    dataset.write_band(chan + 1, pixels, window=window)

        # sync to s3
        if urlparse(self.sink).scheme == 's3':
            local_sink_dir = os.path.dirname(local_sink)
            remote_sink_dir = os.path.dirname(self.sink)  # sic
            sync_dir(local_sink_dir, remote_sink_dir, delete=False)
