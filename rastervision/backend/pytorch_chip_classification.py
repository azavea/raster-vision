from os.path import (join, isfile)
import uuid
import zipfile
import glob
import logging
import json
from subprocess import Popen
import os
import csv
import time
import datetime

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CyclicLR

from rastervision.utils.files import (get_local_path, make_dir, upload_or_copy,
                                      list_paths, download_if_needed,
                                      sync_from_dir, sync_to_dir, str_to_file,
                                      zipdir, file_to_json, json_to_file)
from rastervision.utils.misc import save_img
from rastervision.backend import Backend
from rastervision.data.label import ChipClassificationLabels
from rastervision.utils.misc import terminate_at_exit
from rastervision.backend.torch_utils.chip_classification.plot import plot_xy
from rastervision.backend.torch_utils.chip_classification.data import build_databunch
from rastervision.backend.torch_utils.chip_classification.train import (
    train_epoch, validate_epoch)
from rastervision.backend.torch_utils.chip_classification.model import (
    get_model)

log = logging.getLogger(__name__)


def make_debug_chips(databunch, class_map, tmp_dir, train_uri, max_count=30):
    """Save debug chips for a Databunch for a chip classification dataset

    This saves a plot for each example in the training and validation sets into
    train-debug-chips.zip and valid-debug-chips.zip under the train_uri. This
    is useful for making sure we are feeding correct data into the model.

    Args:
        databunch: DataBunch for chip classification
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
        ds = databunch.train_ds if split == 'train' else databunch.valid_ds
        for i, (x, y) in enumerate(ds):
            if i >= max_count:
                break

            fig, ax = plt.subplots(1)
            plot_xy(ax, x, y, databunch.label_names)
            plt.savefig(
                join(debug_chips_dir, '{}.png'.format(i)), figsize=(6, 6))
            plt.close()

        zipdir(debug_chips_dir, zip_path)
        upload_or_copy(zip_path, zip_uri)

    _make_debug_chips('train')
    _make_debug_chips('valid')


class PyTorchChipClassification(Backend):
    """Chip classification backend using PyTorch and fastai."""

    def __init__(self, task_config, backend_opts, train_opts):
        """Constructor.

        Args:
            task_config: (ChipClassificationConfig)
            backend_opts: (simple_backend_config.BackendOptions)
            train_opts: (pytorch_chip_classification_config.TrainOptions)
        """
        self.task_config = task_config
        self.backend_opts = backend_opts
        self.train_opts = train_opts
        self.inf_learner = None

        torch_cache_dir = '/opt/data/torch-cache'
        os.environ['TORCH_HOME'] = torch_cache_dir

        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log.info('Device = {}'.format(self.device))

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
        valid-img/{class_name}/{scene_id}-{ind}.png

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
            _write_zip(validation_results, 'valid')

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
        batch_size = self.train_opts.batch_size
        chip_size = self.task_config.chip_size
        augmentors = self.train_opts.augmentors
        databunch = build_databunch(
            chip_dir, chip_size, batch_size,
            self.task_config.class_map.get_class_names(),
            self.train_opts.rare_classes, self.train_opts.desired_prob,
            augmentors)

        log.info(databunch)
        num_labels = len(databunch.label_names)
        if self.train_opts.debug:
            make_debug_chips(databunch, self.task_config.class_map, tmp_dir,
                             train_uri)

        # Setup model
        num_labels = len(databunch.label_names)
        model = get_model(
            self.train_opts.model_arch, num_labels, pretrained=True)
        model = model.to(self.device)
        model_path = join(train_dir, 'model')

        # Load weights from a pretrained model.
        pretrained_uri = self.backend_opts.pretrained_uri
        if pretrained_uri:
            log.info('Loading weights from pretrained_uri: {}'.format(
                pretrained_uri))
            pretrained_path = download_if_needed(pretrained_uri, tmp_dir)
            model.load_state_dict(
                torch.load(pretrained_path, map_location=self.device))

        # Possibly resume training from checkpoint.
        start_epoch = 0
        train_state_path = join(train_dir, 'train_state.json')
        if isfile(train_state_path):
            log.info('Resuming from checkpoint: {}\n'.format(model_path))
            train_state = file_to_json(train_state_path)
            start_epoch = train_state['epoch'] + 1
            model.load_state_dict(
                torch.load(model_path, map_location=self.device))

        # Write header of log CSV file.
        metric_names = ['precision', 'recall', 'f1']
        log_path = join(train_dir, 'log.csv')
        if not isfile(log_path):
            with open(log_path, 'w') as log_file:
                log_writer = csv.writer(log_file)
                row = ['epoch', 'time', 'train_loss'] + metric_names
                log_writer.writerow(row)

        # Setup Tensorboard logging.
        if self.train_opts.log_tensorboard:
            log_dir = join(train_dir, 'tb-logs')
            make_dir(log_dir)
            tb_writer = SummaryWriter(log_dir=log_dir)
            if self.train_opts.run_tensorboard:
                log.info('Starting tensorboard process')
                tensorboard_process = Popen(
                    ['tensorboard', '--logdir={}'.format(log_dir)])
                terminate_at_exit(tensorboard_process)

        # Setup optimizer, loss, and LR scheduler.
        loss_fn = torch.nn.CrossEntropyLoss()
        lr = self.train_opts.lr
        opt = optim.Adam(model.parameters(), lr=lr)
        step_scheduler, epoch_scheduler = None, None
        num_epochs = self.train_opts.num_epochs

        if self.train_opts.one_cycle and num_epochs > 1:
            steps_per_epoch = len(databunch.train_ds) // batch_size
            total_steps = num_epochs * steps_per_epoch
            step_size_up = (num_epochs // 2) * steps_per_epoch
            step_size_down = total_steps - step_size_up
            step_scheduler = CyclicLR(
                opt,
                base_lr=lr / 10,
                max_lr=lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                cycle_momentum=False)
            for _ in range(start_epoch * steps_per_epoch):
                step_scheduler.step()

        # Training loop.
        for epoch in range(start_epoch, num_epochs):
            # Train one epoch.
            log.info('-----------------------------------------------------')
            log.info('epoch: {}'.format(epoch))
            start = time.time()
            train_loss = train_epoch(model, self.device, databunch.train_dl,
                                     opt, loss_fn, step_scheduler)
            if epoch_scheduler:
                epoch_scheduler.step()
            log.info('train loss: {}'.format(train_loss))

            # Validate one epoch.
            metrics = validate_epoch(model, self.device, databunch.valid_dl,
                                     num_labels)
            log.info('validation metrics: {}'.format(metrics))

            # Print elapsed time for epoch.
            end = time.time()
            epoch_time = datetime.timedelta(seconds=end - start)
            log.info('epoch elapsed time: {}'.format(epoch_time))

            # Save model and state.
            torch.save(model.state_dict(), model_path)
            train_state = {'epoch': epoch}
            json_to_file(train_state, train_state_path)

            # Append to log CSV file.
            with open(log_path, 'a') as log_file:
                log_writer = csv.writer(log_file)
                row = [epoch, epoch_time, train_loss]
                row += [metrics[k] for k in metric_names]
                log_writer.writerow(row)

            # Write to Tensorboard log.
            if self.train_opts.log_tensorboard:
                for key, val in metrics.items():
                    tb_writer.add_scalar(key, val, epoch)
                tb_writer.add_scalar('train_loss', train_loss, epoch)
                for name, param in model.named_parameters():
                    tb_writer.add_histogram(name, param, epoch)

            if (train_uri.startswith('s3://')
                    and (((epoch + 1) % self.train_opts.sync_interval) == 0)):
                sync_to_dir(train_dir, train_uri)

        # Close Tensorboard.
        if self.train_opts.log_tensorboard:
            tb_writer.close()
            if self.train_opts.run_tensorboard:
                tensorboard_process.terminate()

        # Mark that the command has completed.
        str_to_file('done!', self.backend_opts.train_done_uri)

        # Sync output to cloud.
        sync_to_dir(train_dir, self.backend_opts.train_uri)

    def load_model(self, tmp_dir):
        """Load the model in preparation for one or more prediction calls."""
        if self.model is None:
            model_uri = self.backend_opts.model_uri
            model_path = download_if_needed(model_uri, tmp_dir)

            num_classes = len(self.task_config.class_map)
            model = get_model(
                self.train_opts.model_arch, num_classes, pretrained=False)
            model = model.to(self.device)
            model.load_state_dict(
                torch.load(model_path, map_location=self.device))
            self.model = model

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
        model = self.model.eval()

        with torch.no_grad():
            out = model(chips).cpu()
            labels = ChipClassificationLabels()

            for class_probs, window in zip(out, windows):
                # Add 1 to class_id since they start at 1.
                class_id = int(class_probs.argmax() + 1)
                labels.set_cell(window, class_id, class_probs.numpy())

        return labels
