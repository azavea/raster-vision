from typing import Any, Literal, Optional
from dataclasses import dataclass

from numpy import ndarray

from rastervision.core.box import Box


@dataclass
class DataSample:
    """A chip and labels along with metadata."""
    chip: ndarray
    label: Optional[Any] = None
    split: Optional[Literal['train', 'valid', 'test']] = None
    scene_id: Optional[str] = None
    window: Optional[Box] = None
