from rastervision.core.data import CRSTransformer


class DoubleCRSTransformer(CRSTransformer):
    """Mock CRSTransformer used for testing.

    Assumes pixel coords are 2x map coords.
    """

    def _map_to_pixel(self, map_point):
        return (map_point[0] * 2.0, map_point[1] * 2.0)

    def _pixel_to_map(self, pixel_point):
        return (pixel_point[0] / 2.0, pixel_point[1] / 2.0)
