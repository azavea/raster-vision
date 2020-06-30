from rastervision.pipeline_example_plugin1.sample_pipeline import (
    SamplePipelineConfig)


def get_config(runner, root_uri):
    # The get_config function returns an instantiated PipelineConfig and
    # plays a similar role as a typical "config file" used in other systems.
    # It's different in that it can have loops, conditionals, local variables,
    # etc. The runner argument is the name of the runner used to run the
    # pipeline (eg. local or batch). Any other arguments are passed from the
    # CLI using the -a option.
    names = ['alice', 'bob', 'susan']

    # Note that root_uri is a field that is inherited from PipelineConfig,
    # the parent class of SamplePipelineConfig, and specifies the root URI
    # where any output files are saved.
    return SamplePipelineConfig(root_uri=root_uri, names=names)
