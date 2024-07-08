import logging

from rastervision.pipeline.config import (register_config, Field)
from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.backend import BackendConfig
from rastervision.core.rv_pipeline import RVPipelineConfig
from rastervision.pytorch_learner.learner_config import (
    SolverConfig, ModelConfig, DataConfig, ImageDataConfig, GeoDataConfig)

log = logging.getLogger(__name__)


def pytorch_learner_backend_config_upgrader(cfg_dict: dict,
                                            version: int) -> dict:
    if version == 1:
        # removed in version 2
        cfg_dict.pop('test_mode', None)
    return cfg_dict


@register_config(
    'pytorch_learner_backend',
    upgrader=pytorch_learner_backend_config_upgrader)
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
    save_all_checkpoints: bool = Field(
        False,
        description=(
            'If True, all checkpoints would be saved. The latest checkpoint '
            'would be saved as `last-model.pth`. The checkpoints prior to '
            'last epoch are stored as `model-ckpt-epoch-{N}.pth` where `N` '
            'is the epoch number.'))

    def get_bundle_filenames(self):
        return ['model-bundle.zip']

    def update(self, pipeline: RVPipelineConfig | None = None):
        super().update(pipeline=pipeline)

        if isinstance(self.data, ImageDataConfig):
            if self.data.uri is None and self.data.group_uris is None:
                self.data.uri = pipeline.chip_uri

        if self.data.class_config is None:
            self.data.class_config = pipeline.dataset.class_config

        if not self.data.img_channels:
            self.data.img_channels = self.get_img_channels(pipeline)

    def get_learner_config(self, pipeline: RVPipelineConfig | None):
        raise NotImplementedError()

    def build(self, pipeline: RVPipelineConfig | None, tmp_dir: str):
        raise NotImplementedError()

    def filter_commands(self, commands: list[str]) -> list[str]:
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
