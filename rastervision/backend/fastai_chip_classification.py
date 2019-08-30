from os.path import join, basename, dirname
import uuid
import zipfile
import glob
import logging
import json
from subprocess import Popen

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np

import torch
from fastai.vision import (ImageList, get_transforms, models, cnn_learner)
from fastai.callbacks import TrackEpochCallback
from fastai.basic_train import load_learner

from rastervision.utils.files import (get_local_path, make_dir, upload_or_copy,
                                      list_paths, download_if_needed,
                                      sync_from_dir, sync_to_dir, str_to_file)
from rastervision.utils.misc import save_img
from rastervision.backend import Backend
from rastervision.data.label import ChipClassificationLabels
from rastervision.utils.misc import terminate_at_exit

from rastervision.backend.fastai_utils import (
    SyncCallback, MySaveModelCallback, ExportCallback, MyCSVLogger, Precision,
    Recall, FBeta, zipdir, TensorboardLogger)

log = logging.getLogger(__name__)


def make_debug_chips(data, class_map, tmp_dir, train_uri, max_count=30):
    """Save debug chips for a fastai DataBunch.

    This saves a plot for each example in the training and validation sets into
    train-debug-chips.zip and valid-debug-chips.zip under the train_uri. This
    is useful for making sure we are feeding correct data into the model.

    Args:
        data: fastai DataBunch for a semantic segmentation dataset
        class_map: (rv.ClassMap) class map used to map class ids to colors
        tmp_dir: (str) path to temp directory
        train_uri: (str) URI of root of training output
        max_count: (int) maximum number of chips to generate. If None,
            generates all of them.
    """

    def _make_debug_chips(split):
        debug_chips_dir = join(tmp_dir, '{}-debug-chips'.format(split))
        zip_path = join(tmp_dir, '{}-debug-chips.zip'.format(split))
        zip_uri = join(train_uri, '{}-debug-chips.zip'.format(split))
        make_dir(debug_chips_dir)
        ds = data.train_ds if split == 'train' else data.valid_ds
        for i, (x, y) in enumerate(ds):
            if i >= max_count:
                break

            x.show(y=y)
            plt.savefig(
                join(debug_chips_dir, '{}.png'.format(i)), figsize=(5, 5))
            plt.close()

        zipdir(debug_chips_dir, zip_path)
        upload_or_copy(zip_path, zip_uri)

    _make_debug_chips('train')
    _make_debug_chips('val')


class FastaiChipClassification(Backend):
    """Chip classification backend using PyTorch and fastai."""

    def __init__(self, task_config, backend_opts, train_opts):
        """Constructor.

        Args:
            task_config: (ChipClassificationConfig)
            backend_opts: (simple_backend_config.BackendOptions)
            train_opts: (fastai_chip_classification_config.TrainOptions)
        """
        self.task_config = task_config
        self.backend_opts = backend_opts
        self.train_opts = train_opts
        self.inf_learner = None

    def log_options(self):
        log.info('backend_opts:\n' +
                 json.dumps(self.backend_opts.__dict__, indent=4))
        log.info('train_opts:\n' +
                 json.dumps(self.train_opts.__dict__, indent=4))

    def process_scene_data(self, scene, data, tmp_dir):
        """Make training chips for a scene.

        This writes a set of image chips to {scene_id}/{class_name}/{scene_id}-{ind}.png

        Args:
            scene: (rv.data.Scene)
            data: (rv.data.Dataset)
            tmp_dir: (str) path to temp directory

        Returns:
            (str) path to directory with scene chips {tmp_dir}/{scene_id}
        """
        scene_dir = join(tmp_dir, str(scene.id))

        for ind, (chip, window, labels) in enumerate(data):
            class_id = labels.get_cell_class_id(window)
            # If a chip is not associated with a class, don't
            # use it in training data.
            if class_id is None:
                continue

            class_name = self.task_config.class_map.get_by_id(class_id).name
            class_dir = join(scene_dir, class_name)
            make_dir(class_dir)
            chip_path = join(class_dir, '{}-{}.png'.format(scene.id, ind))
            save_img(chip, chip_path)

        return scene_dir

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """Write zip file with chips for a set of scenes.

        This writes a zip file for a group of scenes at {chip_uri}/{uuid}.zip containing:
        train-img/{class_name}/{scene_id}-{ind}.png
        val-img/{class_name}/{scene_id}-{ind}.png

        This method is called once per instance of the chip command.
        A number of instances of the chip command can run simultaneously to
        process chips in parallel. The uuid in the path above is what allows
        separate instances to avoid overwriting each others' output.

        Args:
            training_results: list of directories generated by process_scene_data
                that all hold training chips
            validation_results: list of directories generated by process_scene_data
                that all hold validation chips
        """
        self.log_options()

        group = str(uuid.uuid4())
        group_uri = join(self.backend_opts.chip_uri, '{}.zip'.format(group))
        group_path = get_local_path(group_uri, tmp_dir)
        make_dir(group_path, use_dirname=True)

        with zipfile.ZipFile(group_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            def _write_zip(scene_dirs, split):
                for scene_dir in scene_dirs:
                    scene_paths = glob.glob(join(scene_dir, '**/*.png'))
                    for path in scene_paths:
                        class_name, fn = path.split('/')[-2:]
                        zipf.write(path, join(split, class_name, fn))

            _write_zip(training_results, 'train')
            _write_zip(validation_results, 'val')

        upload_or_copy(group_path, group_uri)

    def train(self, tmp_dir):
        """Train a model.

        This downloads any previous output saved to the train_uri,
        starts training (or resumes from a checkpoint), periodically
        syncs contents of train_dir to train_uri and after training finishes.

        Args:
            tmp_dir: (str) path to temp directory
        """
        self.log_options()

        # Sync output of previous training run from cloud.
        train_uri = self.backend_opts.train_uri
        train_dir = get_local_path(train_uri, tmp_dir)
        make_dir(train_dir)
        sync_from_dir(train_uri, train_dir)

        # Get zip file for each group, and unzip them into chip_dir.
        chip_dir = join(tmp_dir, 'chips')
        make_dir(chip_dir)
        for zip_uri in list_paths(self.backend_opts.chip_uri, 'zip'):
            zip_path = download_if_needed(zip_uri, tmp_dir)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(chip_dir)

        # Setup data loader.
        size = self.task_config.chip_size
        class_map = self.task_config.class_map
        classes = class_map.get_class_names()
        num_workers = 0 if self.train_opts.debug else 4
        tfms = get_transforms(flip_vert=self.train_opts.flip_vert)

        data = (ImageList.from_folder(chip_dir).split_by_folder(
            train='train', valid='val'))
        train_count = None
        if self.train_opts.train_count is not None:
            train_count = min(len(data.train), self.train_opts.train_count)
        elif self.train_opts.train_prop != 1.0:
            train_count = int(
                round(self.train_opts.train_prop * len(data.train)))
        train_items = data.train.items
        if train_count is not None:
            train_inds = np.random.permutation(np.arange(len(
                data.train)))[0:train_count]
            train_items = train_items[train_inds]
        items = np.concatenate([train_items, data.valid.items])

        data = ImageList(items, chip_dir) \
            .split_by_folder(train='train', valid='val') \
            .label_from_folder(classes=classes) \
            .transform(tfms, size=size) \
            .databunch(bs=self.train_opts.batch_size, num_workers=num_workers)
        log.info(str(data))

        if self.train_opts.debug:
            make_debug_chips(data, class_map, tmp_dir, train_uri)

        # Setup learner.
        ignore_idx = -1
        metrics = [
            Precision(average='weighted', clas_idx=1, ignore_idx=ignore_idx),
            Recall(average='weighted', clas_idx=1, ignore_idx=ignore_idx),
            FBeta(
                average='weighted', clas_idx=1, beta=1, ignore_idx=ignore_idx)
        ]
        model_arch = getattr(models, self.train_opts.model_arch)
        learn = cnn_learner(
            data,
            model_arch,
            metrics=metrics,
            wd=self.train_opts.weight_decay,
            path=train_dir)
        learn.unfreeze()

        if self.train_opts.mixed_prec and torch.cuda.is_available():
            # This loss_scale works for Resnet 34 and 50. You might need to
            # adjust this for other models.
            learn = learn.to_fp16(loss_scale=256)

        # Setup callbacks and train model.
        model_path = get_local_path(self.backend_opts.model_uri, tmp_dir)

        pretrained_uri = self.backend_opts.pretrained_uri
        if pretrained_uri:
            log.info('Loading weights from pretrained_uri: {}'.format(
                pretrained_uri))
            pretrained_path = download_if_needed(pretrained_uri, tmp_dir)
            learn.model.load_state_dict(
                torch.load(pretrained_path, map_location=learn.data.device),
                strict=False)

        # Save every epoch so that resume functionality provided by
        # TrackEpochCallback will work.
        callbacks = [
            TrackEpochCallback(learn),
            MySaveModelCallback(learn, every='epoch'),
            MyCSVLogger(learn, filename='log'),
            ExportCallback(learn, model_path, monitor='f_beta'),
            SyncCallback(train_dir, self.backend_opts.train_uri,
                         self.train_opts.sync_interval)
        ]

        if self.train_opts.log_tensorboard:
            callbacks.append(TensorboardLogger(learn, 'run'))

        if self.train_opts.run_tensorboard:
            log.info('Starting tensorboard process')
            log_dir = join(train_dir, 'logs', 'run')
            tensorboard_process = Popen(
                ['tensorboard', '--logdir={}'.format(log_dir)])
            terminate_at_exit(tensorboard_process)

        lr = self.train_opts.lr
        num_epochs = self.train_opts.num_epochs
        if self.train_opts.one_cycle:
            if lr is None:
                learn.lr_find()
                learn.recorder.plot(suggestion=True, return_fig=True)
                lr = learn.recorder.min_grad_lr
                log.info('lr_find() found lr: {}'.format(lr))
            learn.fit_one_cycle(num_epochs, lr, callbacks=callbacks)
        else:
            learn.fit(num_epochs, lr, callbacks=callbacks)

        if self.train_opts.run_tensorboard:
            tensorboard_process.terminate()

        # Since model is exported every epoch, we need some other way to
        # show that training is finished.
        str_to_file('done!', self.backend_opts.train_done_uri)

        # Sync output to cloud.
        sync_to_dir(train_dir, self.backend_opts.train_uri)

    def load_model(self, tmp_dir):
        """Load the model in preparation for one or more prediction calls."""
        if self.inf_learner is None:
            self.log_options()
            model_uri = self.backend_opts.model_uri
            model_path = download_if_needed(model_uri, tmp_dir)
            self.inf_learner = load_learner(
                dirname(model_path), basename(model_path))
            self.device = torch.device('cuda:0'
                                       if torch.cuda.is_available() else 'cpu')

    def predict(self, chips, windows, tmp_dir):
        """Return predictions for a batch of chips.

        Args:
            chips: (numpy.ndarray) of shape (n, height, width, nb_channels)
                containing a batch of chips
            windows: (List<Box>) windows that are aligned with the chips which
                are aligned with the chips.

        Return:
            (ChipClassificationLabels) containing predictions
        """
        self.load_model(tmp_dir)

        # (batch_size, h, w, nchannels) --> (batch_size, nchannels, h, w)
        chips = torch.Tensor(chips).permute((0, 3, 1, 2)) / 255.
        chips = chips.to(self.device)

        model = self.inf_learner.model.eval()
        preds = model(chips).detach().cpu()

        labels = ChipClassificationLabels()

        for class_probs, window in zip(preds, windows):
            # Add 1 to class_id since they start at 1.
            class_id = int(class_probs.argmax() + 1)
            labels.set_cell(window, class_id, class_probs)

        return labels
