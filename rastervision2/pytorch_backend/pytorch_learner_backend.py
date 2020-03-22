from os.path import join
import tempfile

from rastervision2.pipeline.file_system import (make_dir, upload_or_copy,
                                                zipdir)
from rastervision2.core.backend import Backend, SampleWriter
from rastervision2.core.data_sample import DataSample


class PyTorchLearnerSampleWriter(SampleWriter):
    def __init__(self, output_uri, class_config, tmp_dir_root):
        self.output_uri = output_uri
        self.class_config = class_config
        self.tmp_dir_root = tmp_dir_root

    def __enter__(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory(dir=self.tmp_dir_root)
        self.sample_dir = join(self.tmp_dir_obj.name, 'samples')
        make_dir(self.sample_dir)
        self.sample_ind = 0

        return self

    def __exit__(self, type, value, traceback):
        """
        This writes a zip file for a group of scenes at {chip_uri}/{uuid}.zip.

        This method is called once per instance of the chip command.
        A number of instances of the chip command can run simultaneously to
        process chips in parallel. The uuid in the zip path above is what allows
        separate instances to avoid overwriting each others' output.
        """
        output_path = join(self.tmp_dir_obj.name, 'output.zip')
        zipdir(self.sample_dir, output_path)
        upload_or_copy(output_path, self.output_uri)
        self.tmp_dir_obj.cleanup()

    def write_sample(self, sample: DataSample):
        raise NotImplementedError()


class PyTorchLearnerBackend(Backend):
    def __init__(self, pipeline_cfg, learner_cfg, tmp_dir):
        self.pipeline_cfg = pipeline_cfg
        self.learner_cfg = learner_cfg
        self.tmp_dir = tmp_dir
        self.learner = None

    def train(self):
        learner = self.learner_cfg.build(self.tmp_dir)
        learner.main()

    def load_model(self):
        self.learner = self.learner_cfg.build_from_model_bundle(
            self.learner_cfg.get_model_bundle_uri(), self.tmp_dir)

    def get_sample_writer(self):
        raise NotImplementedError()

    def predict(self, chips, windows):
        raise NotImplementedError()
