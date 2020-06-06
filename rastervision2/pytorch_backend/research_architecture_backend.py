from os.path import join
import tempfile

from rastervision2.pipeline.file_system import (make_dir, upload_or_copy,
                                                zipdir)
from rastervision2.core.backend import Backend, SampleWriter
from rastervision2.core.data_sample import DataSample
from rastervision2.core.data import ClassConfig
from rastervision2.core.rv_pipeline import RVPipelineConfig
from rastervision2.pytorch_learner.learner_config import LearnerConfig
from rastervision2.pytorch_learner.learner import Learner
from rastervision2.pytorch_backend.pytorch_semantic_segmentation import (PyTorchSemanticSegmentation)


class PyTorchLearnerBackend(Backend):

    def __init__(self, pipeline_cfg: RVPipelineConfig,
                 learner_cfg: LearnerConfig, tmp_dir: str):
        # self.pipeline_cfg = pipeline_cfg
        # self.learner_cfg = learner_cfg
        # self.tmp_dir = tmp_dir
        # self.learner = None

    def train(self):
        # learner = self.learner_cfg.build(self.tmp_dir)
        # learner.main()

    def load_model(self):
        # self.learner = Learner.from_model_bundle(
        #     self.learner_cfg.get_model_bundle_uri(), self.tmp_dir)

    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, '{}.zip'.format(
            str(uuid.uuid4())))
        return PyTorchSemanticSegmentationSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

    def predict(self, chips, windows):
        if self.learner is None:
            self.load_model()

        batch_out = self.learner.numpy_predict(chips, raw_out=False)
        labels = SemanticSegmentationLabels()
        for out, window in zip(batch_out, windows):
            labels.set_label_arr(window, out)

        return labels
