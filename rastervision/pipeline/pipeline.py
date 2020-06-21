import logging
from typing import List, TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rastervision.pipeline.pipeline_config import PipelineConfig  # noqa


class Pipeline():
    """A pipeline of commands to run sequentially.

    This is an abstraction over a sequence of commands. Each command is
    represented by a method. This base class has two test commands, and
    new pipelines should be created by subclassing this.

    Note that any split command methods should have the following signature:
    def my_command(self, split_ind: int = 0, num_splits: int = 1)
    The num_splits represents how many parallel jobs should be created, and
    the split_ind is the index of the current job within that set.

    Attributes:
        commands: command names listed in the order in which they should run
        split_commands: names of commands that can be split and run in parallel
        gpu_commands: names of commands that should be executed on GPUs if
            available
    """
    commands: List[str] = ['test_cpu', 'test_gpu']
    split_commands: List[str] = ['test_cpu']
    gpu_commands: List[str] = ['test_gpu']

    def __init__(self, config: 'PipelineConfig', tmp_dir: str):
        """Constructor

        Args:
            config: the configuration of this pipeline
            tmp_dir: the root any temporary directories created by running this
                pipeline
        """
        self.config = config
        self.tmp_dir = tmp_dir

    def test_cpu(self, split_ind: int = 0, num_splits: int = 1):
        """A command to test the ability to run split jobs on CPU."""
        log.info('test_cpu split: {}/{}'.format(split_ind, num_splits))
        log.info(self.config)

    def test_gpu(self):
        """A command to test the ability to run on GPU."""
        log.info(self.config)
