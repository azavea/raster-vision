from typing import TYPE_CHECKING, Dict, Optional

from rastervision.core.data.utils.geojson import buffer_geoms
from rastervision.core.data.vector_transformer import VectorTransformer

if TYPE_CHECKING:
    from rastervision.core.data import CRSTransformer


class BufferTransformer(VectorTransformer):
    """Buffers geometries."""

    def __init__(self,
                 geom_type: str,
                 class_bufs: Optional[Dict[int, Optional[float]]] = None,
                 default_buf: Optional[float] = None):
        """Constructor.

        Args:
            geom_type (str): The geometry type to apply this transform to.
                E.g. "LineString", "Point", "Polygon".
            class_bufs (Dict[int, Optional[float]], optional): Mapping from
                class IDs to buffer amounts (in pixels). If a class ID is not
                found in the mapping, the value specified by the default_buf
                field will be used. If the buffer value for a class is None,
                then no buffering will be applied to the geoms of that
                class.
                Defaults to {}.
            default_buf (Optional[float], optional): Default buffer to apply to
                classes not in class_bufs. If None, no buffering will be
                applied to the geoms of those missing classes. Defaults to
                None.
        """
        self.geom_type = geom_type
        self.class_bufs = class_bufs if class_bufs is not None else {}
        self.default_buf = default_buf

    def transform(self,
                  geojson: dict,
                  crs_transformer: Optional['CRSTransformer'] = None) -> dict:
        return buffer_geoms(
            geojson,
            self.geom_type,
            class_bufs=self.class_bufs,
            default_buf=self.default_buf)
