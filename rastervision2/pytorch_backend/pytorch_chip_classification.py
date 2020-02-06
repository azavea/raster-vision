from os.path import join
import uuid
import tempfile

from rastervision2.pipeline.filesystem import (make_dir, upload_or_copy, zipdir)
from rastervision2.core.data.label import ChipClassificationLabels
from rastervision2.core.backend import Backend, SampleWriter
from rastervision2.core.utils.misc import save_img
from rastervision2.core.data_sample import DataSample


class PyTorchChipClassificationSampleWriter(SampleWriter):
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
        This writes a zip file for a group of scenes at {chip_uri}/{uuid}.zip containing:
        (train|valid)/{class_name}/{scene_id}-{ind}.png

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
        chip_path = join(
            class_dir, '{}-{}.png'.format(sample.scene_id, self.sample_ind))
        save_img(sample.chip, chip_path)
        self.sample_ind += 1


class PyTorchChipClassification(Backend):
    def __init__(self, pipeline_cfg, learner_cfg, tmp_dir):
        self.pipeline_cfg = pipeline_cfg
        self.learner_cfg = learner_cfg
        self.tmp_dir = tmp_dir
        self.learner = None

    def get_sample_writer(self):
        output_uri = join(
            self.pipeline_cfg.chip_uri, '{}.zip'.format(str(uuid.uuid4())))
        return PyTorchChipClassificationSampleWriter(
            output_uri, self.pipeline_cfg.dataset.class_config, self.tmp_dir)

    def train(self):
        learner = self.learner_cfg.build(self.tmp_dir)
        learner.main()

    def load_model(self):
        self.learner = self.learner_cfg.build_from_model_bundle(
            self.learner_cfg.get_model_bundle_uri(), self.tmp_dir)

    def predict(self, chips, windows):
        if self.learner is None:
            self.load_model()

        out = self.learner.numpy_predict(chips, raw_out=True)
        labels = ChipClassificationLabels()

        for class_probs, window in zip(out, windows):
            class_id = class_probs.argmax()
            labels.set_cell(window, class_id, class_probs)

        return labels
