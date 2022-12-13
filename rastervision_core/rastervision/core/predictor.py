from typing import TYPE_CHECKING, List, Optional
from os.path import join
import logging

from rastervision.pipeline import rv_config_ as rv_config
from rastervision.pipeline.config import (build_config, upgrade_config)
from rastervision.pipeline.file_system.utils import (download_if_needed,
                                                     file_to_json, unzip)
from rastervision.core.data.raster_source import ChannelOrderError
from rastervision.core.data import (SemanticSegmentationLabelStoreConfig,
                                    PolygonVectorOutputConfig,
                                    StatsTransformerConfig)
from rastervision.core.analyzer import StatsAnalyzerConfig

if TYPE_CHECKING:
    from rastervision.core.rv_pipeline import RVPipelineConfig
    from rastervision.core.data import SceneConfig

log = logging.getLogger(__name__)


class Predictor():
    """Class for making predictions based off of a model bundle."""

    def __init__(self,
                 model_bundle_uri: str,
                 tmp_dir: str,
                 update_stats: bool = False,
                 channel_order: Optional[List[int]] = None,
                 scene_group: Optional[str] = None):
        """Creates a new Predictor.

        Args:
            model_bundle_uri: URI of the model bundle to use. Can be any
                type of URI that Raster Vision can read.
            tmp_dir: Temporary directory in which to store files that are used
                by the Predictor. This directory is not cleaned up by this
                class.
            channel_order: Option for a new channel order to use for the
                imagery being predicted against. If not present, the
                channel_order from the original configuration in the predict
                package will be used.
        """
        self.tmp_dir = tmp_dir
        self.update_stats = update_stats
        self.model_loaded = False

        bundle_path = download_if_needed(model_bundle_uri)
        bundle_dir = join(tmp_dir, 'bundle')
        unzip(bundle_path, bundle_dir)

        config_path = join(bundle_dir, 'pipeline-config.json')
        config_dict = file_to_json(config_path)
        rv_config.set_everett_config(
            config_overrides=config_dict.get('rv_config'))
        config_dict = upgrade_config(config_dict)
        self.config: 'RVPipelineConfig' = build_config(config_dict)
        self.scene: 'SceneConfig' = self.config.dataset.validation_scenes[0]

        if not hasattr(self.scene.raster_source, 'uris'):
            raise Exception(
                'raster_source in model bundle must have uris as field')

        if not hasattr(self.scene.label_store, 'uri'):
            raise Exception(
                'label_store in model bundle must have uri as field')

        for t in self.scene.raster_source.transformers:
            if isinstance(t, StatsTransformerConfig):
                if scene_group is not None:
                    t.scene_group = scene_group
                else:
                    log.warn(f'Using stats for scene group "{t.scene_group}". '
                             'To use a different scene group, specify '
                             '--scene-group <scene-group-name>.')
            t.update_root(bundle_dir)

        if self.update_stats:
            stats_analyzer = StatsAnalyzerConfig(
                output_uri=join(bundle_dir, 'stats.json'))
            self.config.analyzers = [stats_analyzer]

        self.scene.label_source = None
        self.scene.aoi_uris = None

        self.config.dataset.train_scenes = [self.scene]
        self.config.dataset.validation_scenes = [self.scene]
        self.config.dataset.test_scenes = []
        self.config.train_uri = bundle_dir

        if channel_order is not None:
            self.scene.raster_source.channel_order = channel_order

        self.pipeline = None

    def predict(self, image_uris: List[str], label_uri: str) -> None:
        """Generate predictions for the given image.

        Args:
            image_uris: URIs of the images to make predictions against.
                This can be any type of URI readable by Raster Vision
                FileSystems.
            label_uri: URI to save labels off into
        """
        if self.pipeline is None:
            self.scene.raster_source.uris = image_uris
            self.pipeline = self.config.build(self.tmp_dir)
            if not hasattr(self.pipeline, 'predict'):
                raise Exception(
                    'pipeline in model bundle must have predict method')

        self.scene.raster_source.uris = image_uris
        self.scene.label_store.uri = label_uri

        if isinstance(self.scene.label_store,
                      SemanticSegmentationLabelStoreConfig):
            # create vector outputs for each class without specifying URIs
            self.scene.label_store.vector_output = [
                PolygonVectorOutputConfig(class_id=i)
                for i, _ in enumerate(self.config.dataset.class_config.names)
            ]
            # set URIs
            self.scene.label_store.uri = label_uri
            for vo in self.scene.label_store.vector_output:
                vo.update(self.config, self.scene, uri_prefix=label_uri)

        try:
            if self.update_stats:
                self.pipeline.analyze()
            self.pipeline.predict()
        except ChannelOrderError:
            raise ValueError(
                'The predict package is using a channel_order '
                'with channels unavailable in the imagery.\nTo set a new '
                'channel_order that only uses channels available in the '
                'imagery, use the --channel-order option.')
