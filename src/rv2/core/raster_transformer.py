class RasterTransformer(object):
    def __init__(self, config):
        self.config = config

    def transform(self, chip):
        return chip[:, :, self.config.channel_order]
