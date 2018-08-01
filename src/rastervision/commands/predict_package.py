import tempfile

from rastervision.builders import predict_builder
from rastervision.core.command import Command
from rastervision.core.predict_package import load_predict_package


class PredictPackage(Command):
    def __init__(self,
                 package_zip_uri,
                 labels_uri,
                 image_uris,
                 update_stats=False,
                 channel_order=None):
        self.package_zip_uri = package_zip_uri
        self.labels_uri = labels_uri
        self.image_uris = image_uris
        self.update_stats = update_stats
        self.channel_order = channel_order

    def run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_predict_package(
                self.package_zip_uri,
                temp_dir,
                self.labels_uri,
                self.image_uris,
                update_stats=self.update_stats,
                channel_order=self.channel_order)
            command = predict_builder.build(config)
            command.run()
