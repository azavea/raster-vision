from abc import ABC, abstractmethod


class Labels(ABC):
    '''A set of spatially referenced labels.

    A set of labels predicted by a model or provided by human labelers for the
    sake of training. Every label is associated with a spatial location. For
    object detection, a label is a bounding box surrounding an object and the
    associated class. For classification, a label is a bounding box
    representing a chip within a spatial grid) and the associated class.
    '''
    pass
