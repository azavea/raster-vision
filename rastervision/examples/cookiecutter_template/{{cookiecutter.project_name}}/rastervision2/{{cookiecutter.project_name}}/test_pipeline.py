from typing import List

from rastervision2.pipeline.pipeline import Pipeline


class TestPipeline(Pipeline):
    commands: List[str] = ['print_msg']

    def print_msg(self):
        print(self.config.message)
