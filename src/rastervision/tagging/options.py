from rastervision.common.options import Options


class TaggingOptions(Options):
    def __init__(self, options):
        super().__init__(options)

        if self.aggregate_run_names is None:
            self.use_pretraining = options.get('use_pretraining', False)
            self.target_size = None
