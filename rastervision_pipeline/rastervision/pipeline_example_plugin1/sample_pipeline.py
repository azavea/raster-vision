from typing import List, Optional
from os.path import join

from rastervision.pipeline.pipeline import Pipeline
from rastervision.pipeline.file_system import str_to_file, file_to_str
from rastervision.pipeline.pipeline_config import PipelineConfig
from rastervision.pipeline.config import register_config
from rastervision.pipeline.utils import split_into_groups


# Each Config needs to be registered with a type hint which is used for
# serializing and deserializing to JSON.
@register_config('pipeline_example_plugin1.sample_pipeline')
class SamplePipelineConfig(PipelineConfig):
    # Config classes are configuration schemas. Each field is an attributes
    # with a type and optional default value.
    names: List[str] = ['alice', 'bob']
    message_uris: Optional[List[str]] = None

    def build(self, tmp_dir):
        # The build method is used to instantiate the corresponding object
        # using this configuration.
        return SamplePipeline(self, tmp_dir)

    def update(self):
        # The update method is used to set default values as a function of
        # other values.
        if self.message_uris is None:
            self.message_uris = [
                join(self.root_uri, '{}.txt'.format(name))
                for name in self.names
            ]


class SamplePipeline(Pipeline):
    # The order in which commands run. Each command correspond to a method.
    commands: List[str] = ['save_messages', 'print_messages']

    # Split commands can be split up and run in parallel.
    split_commands = ['save_messages']

    # GPU commands are run using GPUs if available. There are no commands worth running
    # on a GPU in this pipeline.
    gpu_commands = []

    def save_messages(self, split_ind=0, num_splits=1):
        # Save a file for each name with a message.

        # The num_splits is the number of parallel jobs to use and
        # split_ind tracks the index of the parallel job. In this case
        # we are splitting on the names/message_uris.
        split_groups = split_into_groups(
            list(zip(self.config.names, self.config.message_uris)), num_splits)
        split_group = split_groups[split_ind]

        for name, message_uri in split_group:
            message = 'hello {}!'.format(name)
            # str_to_file and most functions in the file_system package can
            # read and write transparently to different file systems based on
            # the URI pattern.
            str_to_file(message, message_uri)
            print('Saved message to {}'.format(message_uri))

    def print_messages(self):
        # Read all the message files and print them.
        for message_uri in self.config.message_uris:
            message = file_to_str(message_uri)
            print(message)
