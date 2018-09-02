import zipfile

from google.protobuf import json_format

from rastervision.utils.files import (download_if_needed,
                                      make_dir)
from rastervision.protos.command_pb2 import CommandConfig as CommandConfigMsg

class Predictor():
    """Class for making predictions based off of a prediction package"""
    def __init__(self,
                 prediction_package_uri,
                 tmp_dir,
                 update_stats=False,
                 channel_order=None):
        self.tmp_dir = tmp_dir
        self.update_stats = update_stats
        self.model_loaded = False

        package_zip_path = download_if_needed(prediction_package_uri, tmp_dir)
        package_dir = os.path.join(tmp_dir, 'package')
        make_dir(package_dir)
        with zipfile.ZipFile(package_zip_path, 'r') as package_zip:
            package_zip.extractall(path=package_dir)

        # Read bundle command config
        with open(os.path.join(package_dir, "bundle.json")) as f:
            bundle_config_json = f.read()
        bundle_config = json_format.ParseJson(bundle_config_json, CommandConfigMsg.BundleConfig())

        self.task_config = rv.TaskConfig.from_proto(bundle_config.task) \
                                   .load_bundle_files(package_dir)
        self.backend_config = rv.BackendConfig.from_proto(bundle_config.backend) \
                                         .load_bundle_files(package_dir)
        scene_builder = rv.SceneConfig.from_proto(bundle_config.scene) \
                                      .load_bundle_files(package_dir) \
                                      .to_builder() \
                                      .with_scene_id("PREDICTOR")

        if channel_order:
            scene_builder = scene_builder.with_channel_order(channel_order)

        self.scene_config = scene_builder.build()

        self.analyzer_configs = []
        if update_stats:
            for analyzer in bundle_config.analyzers:
                a = rv.AnalyzerConfig.from_proto(analyzer) \
                                     .load_bundle_files(package_dir)
                self.analyzer_configs.append(a)

    def load_model(self):
        self.backend = self.backend_config.create_backend(self.task_config)
        self.backend.load_model(self.tmp_dir)
        self.task = self.task_config.create_task(self.backend)
        self.analyzers = []
        for analyzer_config in self.analyzer_configs:
            self.analyzers.append(analyzer_config.create_analyzer())
        self.model_loaded = True

    def predict(self, image_uri, label_uri=None):
        if not self.model_loaded:
            self.load_model()
        scene_config = self.scene_config.for_prediction(image_uri, label_uri) \
                                        .create_local(tmp_dir)

        scene = scene_config.create_scene(self.task_config, self.tmp_dir)
        # If we are analyzing per scene, run analyzers
        # Analyzers should overwrite files in the tmp_dir
        if self.update_stats:
            for analyzer in self.analyzers:
                analyzer.process([scene])

            # Reload scene to refresh any new stats
            scene = scene_config.create_scene(self.task_config, self.tmp_dir)

        labels = self.task.predict_scene(scene, self.tmp_dir)
        if scene.prediction_label_store:
            scene.prediction_label_store.save(labels)
        return labels


if __name__ == "__main__":
    pass
