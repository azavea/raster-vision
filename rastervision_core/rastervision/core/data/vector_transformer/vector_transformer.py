from abc import (ABC, abstractmethod)


class VectorTransformer(ABC):
    """Transforms vector data."""

    def __call__(self, geojson: dict) -> dict:
        return self.transform(geojson)

    @abstractmethod
    def transform(self, geojson: dict) -> dict:
        """Transform a GeoJSON mapping of vector data.

        Args:
            geojson (dict): A GeoJSON-like mapping of a FeatureCollection.

        Returns:
            dict: Transformed GeoJSON.
        """
        pass
