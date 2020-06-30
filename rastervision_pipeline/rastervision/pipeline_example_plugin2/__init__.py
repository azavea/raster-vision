# flake8: noqa

# Must import pipeline package first.
import rastervision.pipeline

# Then import any modules that add Configs so that the register_config decorators
# get called.
import rastervision.pipeline_example_plugin2.deluxe_message_maker


def register_plugin(registry):
    # Can be used to manually update the registry. Useful
    # for adding new FileSystems and Runners.
    pass
