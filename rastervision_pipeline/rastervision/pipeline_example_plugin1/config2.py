from rastervision.pipeline_example_plugin1.sample_pipeline2 import (
    SamplePipeline2Config, MessageMakerConfig)


def get_config(runner, root_uri):
    names = ['alice', 'bob', 'susan']
    # Same as before except we can set the greeting to be
    # 'hola' instead of 'hello'.
    message_maker = MessageMakerConfig(greeting='hola')
    return SamplePipeline2Config(
        root_uri=root_uri, names=names, message_maker=message_maker)
