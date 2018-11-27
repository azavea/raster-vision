import os
import zipfile
import logging

import click
from google.protobuf import json_format

from rastervision.command import Command
from rastervision.utils.files import (upload_or_copy, make_dir)

log = logging.getLogger(__name__)


class BundleCommand(Command):
    """Bundles all the necessary files together into a prediction package."""

    def __init__(self, command_config):
        self.command_config = command_config

    def run(self, tmp_dir=None):
        if not tmp_dir:
            tmp_dir = self.get_tmp_dir()

        cc = self.command_config

        if not cc.task.predict_package_uri:
            msg = 'Skipping bundling of prediction package, no URI is set...'.format(
                cc.task.predict_package_uri)
            click.echo(click.style(msg, fg='yellow'))
            return

        msg = 'Bundling prediction package to {}...'.format(
            cc.task.predict_package_uri)
        log.info(msg)

        bundle_dir = os.path.join(tmp_dir, 'bundle')
        make_dir(bundle_dir)
        package_path = os.path.join(tmp_dir, 'predict_package.zip')
        bundle_files = []
        new_task, task_files = cc.task.save_bundle_files(bundle_dir)
        bundle_files.extend(task_files)
        new_backend, backend_files = cc.backend.save_bundle_files(bundle_dir)
        bundle_files.extend(backend_files)
        new_scene, scene_files = cc.scene.save_bundle_files(bundle_dir)
        bundle_files.extend(scene_files)
        new_analyzers = []
        for analyzer in cc.analyzers:
            new_analyzer, analyzer_files = analyzer.save_bundle_files(
                bundle_dir)
            new_analyzers.append(new_analyzer)
            bundle_files.extend(analyzer_files)

        new_bundle_config = cc.to_builder() \
                              .with_task(new_task) \
                              .with_backend(new_backend) \
                              .with_scene(new_scene) \
                              .with_analyzers(new_analyzers) \
                              .build()

        # Save bundle command config
        bundle_config_path = os.path.join(tmp_dir, 'bundle_config.json')
        bundle_json = json_format.MessageToJson(new_bundle_config.to_proto())
        with open(bundle_config_path, 'w') as f:
            f.write(bundle_json)

        with zipfile.ZipFile(package_path, 'w') as package_zip:
            for path in bundle_files:
                package_zip.write(path, arcname=os.path.basename(path))
            package_zip.write(
                bundle_config_path,
                arcname=os.path.basename(bundle_config_path))

        upload_or_copy(package_path, cc.task.predict_package_uri)
