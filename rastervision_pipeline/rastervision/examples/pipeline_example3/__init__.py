# flake8: noqa

# Always need to import first.
import rastervision.pipeline

# Need to import any modules with register_config decorators.
import rastervision.examples.deluxe_message_maker.deluxe_message_maker


def register_plugin(registry):
    # Can be used to manually update the registry. Useful
    # for adding new FileSystems and Runners.
    pass
