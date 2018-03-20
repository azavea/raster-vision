from abc import ABC, abstractmethod


class Labels(ABC):
    '''A set of spatially referenced labels.

    A set of labels associated with their location predicted by a model or
    provided by human labelers for the sake of training. An example of an label
    is a bounding box in the case of object detection or a raster in case of
    segmentation.
    '''
    pass
