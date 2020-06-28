from os.path import join
import uuid

from rastervision.pipeline.file_system import (make_dir)
from rastervision.core.data.label import ChipClassificationLabels
from rastervision.core.utils.misc import save_img
from rastervision.core.data_sample import DataSample
from rastervision.pytorch_backend.pytorch_learner_backend import (
    PyTorchLearnerSampleWriter, PyTorchLearnerBackend)


class PyTorchChipClassificationSampleWriter(PyTorchLearnerSampleWriter):
    def write_sample(self, sample: DataSample):
        """
        This writes a training or validation sample to
        (train|valid)/{class_name}/{scene_id}-{ind}.png
        """
        class_id = sample.labels.get_cell_class_id(sample.window)
        # If a chip is not associated with a class, don't
        # use it in training data.
        if class_id is None:
            return

        split_name = 'train' if sample.is_train else 'valid'
        class_name = self.class_config.names[class_id]
        class_dir = join(self.sample_dir, split_name, class_name)
        make_dir(class_dir)
        chip_path = join(class_dir, '{}-{}.png'.format(sample.scene_id,
                                                       self.sample_ind))
        save_img(sample.chip, chip_path)
        self.sample_ind += 1


class PyTorchChipClassification(PyTorchLearnerBackend):
    def get_sample_writer(self):
        output_uri = join(self.pipeline_cfg.chip_uri, '{}.zip'.format(
            str(uuid.uuid4())))
        return PyTorchChipClassificationSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

    def predict(self, chips, windows):
        if self.learner is None:
            self.load_model()

        out = self.learner.numpy_predict(chips, raw_out=True)
        labels = ChipClassificationLabels()

        for class_probs, window in zip(out, windows):
            class_id = class_probs.argmax()
            labels.set_cell(window, class_id, class_probs)

        return labels
