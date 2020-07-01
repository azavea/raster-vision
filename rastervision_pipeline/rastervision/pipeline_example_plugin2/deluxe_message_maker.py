from rastervision.pipeline.config import register_config
from rastervision.pipeline_example_plugin1.sample_pipeline2 import (
    MessageMakerConfig, MessageMaker)


# You always need to use the register_config decorator.
@register_config('pipeline_example_plugin2.deluxe_message_maker')
class DeluxeMessageMakerConfig(MessageMakerConfig):
    # Note that this inherits the greeting field from MessageMakerConfig.
    level: int = 1

    def build(self):
        return DeluxeMessageMaker(self)


class DeluxeMessageMaker(MessageMaker):
    def make_message(self, name):
        # Uses the level field to determine the number of exclamation marks.
        exclamation_marks = '!' * self.config.level
        return '{} {}{}'.format(self.config.greeting, name, exclamation_marks)
