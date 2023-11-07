from abc import (ABC, abstractmethod)
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rastervision.core.data import CRSTransformer


class VectorTransformer(ABC):
    """Transforms vector data."""

    def __call__(self,
                 geojson: dict,
                 crs_transformer: Optional['CRSTransformer'] = None,
                 **kwargs) -> dict:
        """Shortcut for :meth:`.transform`.

        Args:
            geojson (dict): A GeoJSON-like mapping of a FeatureCollection.
            crs_transformer (Optional[CRSTransformer]): CRSTransformer.
                Defaults to None.
            **kwargs: Extra args for :meth:`.transform`.

        Returns:
            dict: Transformed GeoJSON.
        """
        return self.transform(
            geojson, crs_transformer=crs_transformer, **kwargs)

    @abstractmethod
    def transform(self,
                  geojson: dict,
                  crs_transformer: Optional['CRSTransformer'] = None) -> dict:
        """Transform a GeoJSON mapping of vector data.

        Args:
            geojson (dict): A GeoJSON-like mapping of a FeatureCollection.

        Returns:
            dict: Transformed GeoJSON.
        """
