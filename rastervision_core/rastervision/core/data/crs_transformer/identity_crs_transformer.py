from rastervision.core.data.crs_transformer import CRSTransformer


class IdentityCRSTransformer(CRSTransformer):
    """Transformer for when map coordinates are already in pixel coordinates.

    This is useful for non-georeferenced imagery.
    """

    def _map_to_pixel(self, map_point):
        """Identity function.

        Args:
            map_point: (x, y) tuple in pixel coordinates

        Returns:
            (x, y) tuple in pixel coordinates
        """
        return map_point

    def _pixel_to_map(self, pixel_point):
        """Identity function.

        Args:
            pixel_point: (x, y) tuple in pixel coordinates

        Returns:
            (x, y) tuple in pixel coordinates
        """
        return pixel_point
