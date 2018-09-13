import os
import zipfile

from google.protobuf import json_format

from rastervision.command import Command
from rastervision.utils.files import (upload_or_copy, make_dir)


class BundleCommand(Command):
    """Bundles all the necessary files together into a prediction package."""

    def __init__(self, bundle_config, task_config, backend_config,
                 scene_config, analyzer_configs):
        self.bundle_config = bundle_config
        self.task_config = task_config
        self.backend_config = backend_config
        self.scene_config = scene_config
        self.analyzer_configs = analyzer_configs

    def run(self, tmp_dir):
        bundle_dir = os.path.join(tmp_dir, "bundle")
        make_dir(bundle_dir)
        package_path = os.path.join(tmp_dir, "predict_package.zip")
        bundle_files = []
        bundle_files.extend(self.task_config.save_bundle_files(bundle_dir))
        bundle_files.extend(self.backend_config.save_bundle_files(bundle_dir))
        bundle_files.extend(self.scene_config.save_bundle_files(bundle_dir))
        for analyzer in self.analyzer_configs:
            bundle_files.extend(analyzer.save_bundle_files(bundle_dir))

        # Save bundle command config
        bundle_config_path = os.path.join(tmp_dir, "bundle_config.json")
        bundle_json = json_format.MessageToJson(self.bundle_config.to_proto())
        with open(bundle_config_path, 'w') as f:
            f.write(bundle_json)

        with zipfile.ZipFile(package_path, 'w') as package_zip:
            for path in bundle_files:
                package_zip.write(path, arcname=os.path.basename(path))
            package_zip.write(
                bundle_config_path,
                arcname=os.path.basename(bundle_config_path))

        upload_or_copy(package_path, self.task_config.predict_package_uri)
