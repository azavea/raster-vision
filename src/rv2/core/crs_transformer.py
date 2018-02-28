import pyproj


class CRSTransformer(object):
    def __init__(self, image_dataset):
        self.image_dataset = image_dataset
        self.web_proj = pyproj.Proj(init='epsg:4326')
        image_crs = image_dataset.crs['init']
        self.image_proj = pyproj.Proj(init=image_crs)

    def web_to_pixel(self, web_point):
        # Convert point in Web Mercator to pixel coords for
        # image_dataset.
        image_point = pyproj.transform(
            self.web_proj, self.image_proj, web_point[0], web_point[1])
        pixel_point = self.image_dataset.index(image_point[0], image_point[1])
        return pixel_point

    def pixel_to_web(self, pixel_point):
        # Convert point in pixel coords for image_dataset to
        # Web Mercator.
        image_point = self.image_dataset.ul(
            int(pixel_point[0]), int(pixel_point[1]))
        web_point = pyproj.transform(
            self.image_proj, self.web_proj, image_point[0], image_point[1])
        return web_point
