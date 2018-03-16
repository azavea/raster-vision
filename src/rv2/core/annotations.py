from abc import ABC, abstractmethod


class Annotations(ABC):
    '''A set of spatially referenced annotations.

    A set of annotations associated with their location in 
    predicted by a model or provided by human annotators for the sake of
    training.An example of an annotation is a bounding box in the case of object
    detection or a raster in case of segmentation.
    '''
    pass
