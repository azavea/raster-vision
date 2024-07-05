from rastervision.pipeline.pipeline import Pipeline


class TestPipeline(Pipeline):
    commands: list[str] = ['print_msg']

    def print_msg(self):
        print(self.config.message)
