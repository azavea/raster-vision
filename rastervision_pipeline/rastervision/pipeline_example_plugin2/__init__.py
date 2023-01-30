# flake8: noqa


def register_plugin(registry):
    """Each plugin must register itself and FileSystems, Runners it defines.
    
    The version number helps ensure backward compatibility of configs across
    versions. If you change the fields of a config but want it to remain
    backward-compatible you can increment the version below and define a
    config-upgrader function that makes the old version of the config dict
    compatible with the new version. This upgrader function should be passed to
    the :func:`.register_config` decorator of the config in question.
    """
    registry.set_plugin_version('rastervision.pipeline_example_plugin2', 0)


# Must import pipeline package first.
import rastervision.pipeline

# Then import any modules that add Configs so that the register_config decorators
# get called.
import rastervision.pipeline_example_plugin2.deluxe_message_maker
