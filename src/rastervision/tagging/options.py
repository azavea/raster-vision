from rastervision.common.options import Options


class TaggingOptions(Options):
    def __init__(self, options):
        super().__init__(options)

        self.use_pretraining = options['use_pretraining']
        self.target_size = None
