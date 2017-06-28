from rastervision.common.options import Options


class TaggingOptions(Options):
    def __init__(self, options):
        super().__init__(options)

        self.active_tags = options.get('active_tags')
        self.use_pretraining = options.get('use_pretraining', False)
        self.freeze_base = options.get('freeze_base', False)
        self.target_size = None
        self.active_tags_prob = options.get('active_tags_prob')
