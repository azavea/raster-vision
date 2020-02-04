from pydantic import BaseModel
from numpy import ndarray

from rastervision.v2.rv.box import Box
from rastervision.v2.rv.data import Labels


class DataSample(BaseModel):
    chip: ndarray
    window: Box
    labels: Labels
    scene_id: str = 'default'
    is_train: bool = True

    class Config:
        arbitrary_types_allowed = True
