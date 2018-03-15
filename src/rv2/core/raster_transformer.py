class RasterTransformer(object):
    """Transforms chips according to a config."""

    def __init__(self, options):
        """Construct a new RasterTransformer.

        Args:
            options: RasterTransformerConfig
        """
        self.options = options

    def transform(self, chip):
        """Transform a chip.

        Selects a subset of the channels.

        Args:
            chip: [height, width, channels] numpy array

        Returns:
            [height, width, channels] numpy array where channels is equal
                to len(self.options.channel_order)
        """
        return chip[:, :, self.options.channel_order]
