class RasterTransformer(object):
    """A mechanism for transforming chips according to a config."""
    def __init__(self, config):
        self.config = config

    def transform(self, chip):
        return chip[:, :, self.config.channel_order]
