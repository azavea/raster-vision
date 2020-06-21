from pydantic import BaseModel
from numpy import ndarray

from rastervision.core.box import Box
from rastervision.core.data import Labels


class DataSample(BaseModel):
    """A chip and labels along with metadata."""
    chip: ndarray
    window: Box
    labels: Labels
    scene_id: str = 'default'
    is_train: bool = True

    class Config:
        arbitrary_types_allowed = True
