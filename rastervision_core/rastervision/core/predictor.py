from os.path import join
import zipfile
import logging

from rastervision.pipeline import rv_config
from rastervision.pipeline.config import (build_config, upgrade_config)
from rastervision.pipeline.file_system.utils import (download_if_needed,
                                                     make_dir, file_to_json)
from rastervision.core.data.raster_source import ChannelOrderError
from rastervision.core.analyzer import StatsAnalyzerConfig

log = logging.getLogger(__name__)


class Predictor():
    """Class for making predictions based off of a model bundle."""

    def __init__(self,
                 model_bundle_uri,
                 tmp_dir,
                 update_stats=False,
                 channel_order=None):
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

        bundle_path = download_if_needed(model_bundle_uri, tmp_dir)
        bundle_dir = join(tmp_dir, 'bundle')
        make_dir(bundle_dir)
        with zipfile.ZipFile(bundle_path, 'r') as bundle_zip:
            bundle_zip.extractall(path=bundle_dir)

        config_path = join(bundle_dir, 'pipeline-config.json')
        config_dict = file_to_json(config_path)
        rv_config.set_everett_config(
            config_overrides=config_dict.get('rv_config'))
        config_dict = upgrade_config(config_dict)

        self.pipeline = build_config(config_dict).build(tmp_dir)
        self.scene = None

        if not hasattr(self.pipeline, 'predict'):
            raise Exception(
                'pipeline in model bundle must have predict method')

        self.scene = self.pipeline.config.dataset.validation_scenes[0]

        if not hasattr(self.scene.raster_source, 'uris'):
            raise Exception(
                'raster_source in model bundle must have uris as field')

        if not hasattr(self.scene.label_store, 'uri'):
            raise Exception(
                'label_store in model bundle must have uri as field')

        for t in self.scene.raster_source.transformers:
            t.update_root(bundle_dir)

        if self.update_stats:
            stats_analyzer = StatsAnalyzerConfig(
                output_uri=join(bundle_dir, 'stats.json'))
            self.pipeline.config.analyzers = [stats_analyzer]

        self.scene.label_source = None
        self.scene.aoi_uris = None
        self.pipeline.config.dataset.train_scenes = [self.scene]
        self.pipeline.config.dataset.validation_scenes = [self.scene]
        self.pipeline.config.dataset.test_scenes = None
        self.pipeline.config.train_uri = bundle_dir

        if channel_order is not None:
            self.scene.raster_source.channel_order = channel_order

    def predict(self, image_uris, label_uri):
        """Generate predictions for the given image.

        Args:
            image_uris: URIs of the images to make predictions against.
                This can be any type of URI readable by Raster Vision
                FileSystems.
            label_uri: URI to save labels off into.
        """
        try:
            self.scene.raster_source.uris = image_uris
            self.scene.label_store.uri = label_uri
            if self.update_stats:
                self.pipeline.analyze()
            self.pipeline.predict()
        except ChannelOrderError:
            raise ValueError(
                'The predict package is using a channel_order '
                'with channels unavailable in the imagery.\nTo set a new '
                'channel_order that only uses channels available in the '
                'imagery, use the --channel-order option.')
