from typing import Optional, List
import logging

from rastervision.pipeline.config import (register_config, Field)
from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.backend import BackendConfig
from rastervision.core.rv_pipeline import RVPipelineConfig
from rastervision.pytorch_learner.learner_config import (
    SolverConfig, ModelConfig, DataConfig, ImageDataConfig, GeoDataConfig)

log = logging.getLogger(__name__)


@register_config('pytorch_learner_backend')
class PyTorchLearnerBackendConfig(BackendConfig):
    """Configure a :class:`.PyTorchLearnerBackend`."""

    model: ModelConfig
    solver: SolverConfig
    data: DataConfig
    log_tensorboard: bool = Field(
        True, description='If True, log events to Tensorboard log files.')
    run_tensorboard: bool = Field(
        False,
        description='If True, run Tensorboard server pointing at log files.')
    test_mode: bool = Field(
        False,
        description=
        ('This field is passed along to the LearnerConfig which is returned by '
         'get_learner_config(). For more info, see the docs for'
         'pytorch_learner.learner_config.LearnerConfig.test_mode.'))

    def get_bundle_filenames(self):
        return ['model-bundle.zip']

    def update(self, pipeline: Optional[RVPipelineConfig] = None):
        super().update(pipeline=pipeline)

        if isinstance(self.data, ImageDataConfig):
            if self.data.uri is None and self.data.group_uris is None:
                self.data.uri = pipeline.chip_uri

        if not self.data.class_names:
            # We want to defer validating class_names against class_colors
            # until we have updated both. Hence, we use Config.copy(update=)
            # here because it does not trigger pydantic validators.
            self.data = self.data.copy(
                update={'class_names': pipeline.dataset.class_config.names})
        if not self.data.class_colors:
            self.data.class_colors = pipeline.dataset.class_config.colors

        if not self.data.img_channels:
            self.data.img_channels = self.get_img_channels(pipeline)

    def get_learner_config(self, pipeline: Optional[RVPipelineConfig]):
        raise NotImplementedError()

    def build(self, pipeline: Optional[RVPipelineConfig], tmp_dir: str):
        raise NotImplementedError()

    def filter_commands(self, commands: List[str]) -> List[str]:
        nochip = isinstance(self.data, GeoDataConfig)
        if nochip and 'chip' in commands:
            commands = [c for c in commands if c != 'chip']
        return commands

    def get_img_channels(self, pipeline_cfg: RVPipelineConfig) -> int:
        """Determine img_channels from scenes."""
        all_scenes = pipeline_cfg.dataset.all_scenes
        if len(all_scenes) == 0:
            return 3
        for scene_cfg in all_scenes:
            if scene_cfg.raster_source.channel_order is not None:
                return len(scene_cfg.raster_source.channel_order)
        log.info(
            'Could not determine number of image channels from '
            'DataConfig.img_channels or RasterSourceConfig.channel_order. '
            'Building first scene to figure it out. This might take some '
            'time. To avoid this, specify one of the above.')
        with get_tmp_dir() as tmp_dir:
            scene = all_scenes[0].build(
                pipeline_cfg.dataset.class_config,
                tmp_dir,
                use_transformers=True)
            img_channels = scene.raster_source.num_channels
        return img_channels
