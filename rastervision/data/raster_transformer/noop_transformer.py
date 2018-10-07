from rastervision.data.raster_transformer.raster_transformer \
    import RasterTransformer


class NoopTransformer(RasterTransformer):
    """No-op transformer
    """

    def __init__(self):
        pass

    def transform(self, chip, channel_order=None):
        return chip
