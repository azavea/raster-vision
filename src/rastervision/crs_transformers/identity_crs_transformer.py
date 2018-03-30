from rastervision.core.crs_transformer import CRSTransformer


class IdentityCRSTransformer(CRSTransformer):
    def web_to_pixel(self, web_point):
        return web_point

    def pixel_to_web(self, pixel_point):
        return pixel_point
