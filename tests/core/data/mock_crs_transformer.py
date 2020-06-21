from rastervision.core.data import CRSTransformer


class DoubleCRSTransformer(CRSTransformer):
    """Mock CRSTransformer used for testing.

    Assumes map coords are 2x pixels coords.
    """

    def map_to_pixel(self, web_point):
        return (web_point[0] * 2.0, web_point[1] * 2.0)

    def pixel_to_map(self, pixel_point):
        return (pixel_point[0] / 2.0, pixel_point[1] / 2.0)
