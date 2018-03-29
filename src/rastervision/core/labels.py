from abc import ABC, abstractmethod


class Labels(ABC):
    """A set of spatially referenced labels for a chip or RasterSource.

    A set of labels predicted by a model or provided by human labelers for the
    sake of training. Every label is associated with a spatial location and a
    class. For object detection, a label is a bounding box surrounding an
    object and the associated class. For classification, a label is a bounding
    box representing a cell/chip within a spatial grid and its class.
    For segmentation, a label is a pixel and its class.
    """
    pass
