# flake8: noqa

# Must import pipeline package first.
import rastervision.pipeline

# Then import any modules that add Configs so that the register_config decorators
# get called.
import rastervision.pipeline_example.sample_pipeline
import rastervision.pipeline_example.sample_pipeline2
import rastervision.pipeline_example.deluxe_message_maker

def register_plugin(registry):
    pass
