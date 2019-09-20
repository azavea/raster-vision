import os
from os.path import join, basename, isfile
import uuid
import zipfile
import glob
import logging
import json
import csv
import time
import datetime
from subprocess import Popen

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from rastervision.utils.files import (get_local_path, make_dir, upload_or_copy,
                                      list_paths, download_if_needed,
                                      sync_from_dir, sync_to_dir, str_to_file,
                                      json_to_file, file_to_json, zipdir)
from rastervision.utils.misc import save_img, terminate_at_exit
from rastervision.backend import Backend
from rastervision.data import ObjectDetectionLabels
from rastervision.backend.torch_utils.object_detection.data import build_databunch
from rastervision.backend.torch_utils.object_detection.plot import plot_xy
from rastervision.backend.torch_utils.object_detection.model import MyFasterRCNN
from rastervision.backend.torch_utils.object_detection.train import (
    train_epoch, validate_epoch)
from rastervision.backend.torch_utils.scheduler import (CyclicLR)

log = logging.getLogger(__name__)


def make_debug_chips(databunch, class_map, tmp_dir, train_uri, max_count=30):
    """Save debug chips for a DataBunch.

    This saves a plot for each example in the training and validation sets into
    train-debug-chips.zip and valid-debug-chips.zip under the train_uri. This
    is useful for making sure we are feeding correct data into the model.

    Args:
        data: DataBunch for an object detection problem
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
            plot_xy(ax, x, y, ds.label_names)
            plt.savefig(
                join(debug_chips_dir, '{}.png'.format(i)), figsize=(6, 6))
            plt.close()

        zipdir(debug_chips_dir, zip_path)
        upload_or_copy(zip_path, zip_uri)

    _make_debug_chips('train')
    _make_debug_chips('val')


class PyTorchObjectDetection(Backend):
    """Object detection using PyTorch and Faster-RCNN/Resnet50 from torchvision."""

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

        # Setup caching for torchvision pretrained models.
        torch_cache_dir = '/opt/data/torch-cache'
        os.environ['TORCH_HOME'] = torch_cache_dir

        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def log_options(self):
        log.info('backend_opts:\n' +
                 json.dumps(self.backend_opts.__dict__, indent=4))
        log.info('train_opts:\n' +
                 json.dumps(self.train_opts.__dict__, indent=4))

    def process_scene_data(self, scene, data, tmp_dir):
        """Process each scene's training data.

        This writes {scene_id}/{scene_id}-{ind}.png and
        {scene_id}/{scene_id}-labels.json in COCO format.

        Args:
            scene: Scene
            data: TrainingData

        Returns:
            backend-specific data-structures consumed by backend's
            process_sceneset_results
        """
        scene_dir = join(tmp_dir, str(scene.id))
        labels_path = join(scene_dir, '{}-labels.json'.format(scene.id))

        make_dir(scene_dir)
        images = []
        annotations = []
        categories = [{
            'id': item.id,
            'name': item.name
        } for item in self.task_config.class_map.get_items()]

        for im_ind, (chip, window, labels) in enumerate(data):
            im_id = '{}-{}'.format(scene.id, im_ind)
            fn = '{}.png'.format(im_id)
            chip_path = join(scene_dir, fn)
            save_img(chip, chip_path)
            images.append({
                'file_name': fn,
                'id': im_id,
                'height': chip.shape[0],
                'width': chip.shape[1]
            })

            npboxes = labels.get_npboxes()
            npboxes = ObjectDetectionLabels.global_to_local(npboxes, window)
            for box_ind, (box, class_id) in enumerate(
                    zip(npboxes, labels.get_class_ids())):
                bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
                bbox = [int(i) for i in bbox]
                annotations.append({
                    'id': '{}-{}'.format(im_id, box_ind),
                    'image_id': im_id,
                    'bbox': bbox,
                    'category_id': int(class_id)
                })

        coco_dict = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        json_to_file(coco_dict, labels_path)

        return scene_dir

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """After all scenes have been processed, process the result set.

        This writes a zip file for a group of scenes at {chip_uri}/{uuid}.zip
        containing:
        train/{scene_id}-{ind}.png
        train/{scene_id}-labels.json
        val/{scene_id}-{ind}.png
        val/{scene_id}-labels.json

        Args:
            training_results: dependent on the ml_backend's process_scene_data
            validation_results: dependent on the ml_backend's
                process_scene_data
        """
        self.log_options()

        group = str(uuid.uuid4())
        group_uri = join(self.backend_opts.chip_uri, '{}.zip'.format(group))
        group_path = get_local_path(group_uri, tmp_dir)
        make_dir(group_path, use_dirname=True)

        with zipfile.ZipFile(group_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            def _write_zip(results, split):
                for scene_dir in results:
                    scene_paths = glob.glob(join(scene_dir, '*'))
                    for p in scene_paths:
                        zipf.write(p, join(split, basename(p)))

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

        # Setup dataset and dataloaders.
        batch_size = self.train_opts.batch_size
        chip_size = self.task_config.chip_size
        databunch = build_databunch(chip_dir, chip_size, batch_size)
        num_labels = len(databunch.label_names)
        if self.train_opts.debug:
            make_debug_chips(databunch, self.task_config.class_map, tmp_dir,
                             train_uri)

        # Setup model
        num_labels = len(databunch.label_names)
        model = MyFasterRCNN(
            self.train_opts.model_arch, num_labels, chip_size, pretrained=True)
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
        log_path = join(train_dir, 'log.csv')
        if not isfile(log_path):
            with open(log_path, 'w') as log_file:
                log_writer = csv.writer(log_file)
                row = ['epoch'] + ['map50', 'time'] + model.subloss_names
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

        # Setup optimizer.
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
            start = time.time()

            # Train one epoch.
            train_loss = train_epoch(model, self.device, databunch.train_dl,
                                     opt, step_scheduler, epoch_scheduler)
            if epoch_scheduler:
                epoch_scheduler.step()
            log.info('----------------------------------------')
            log.info('epoch: {}'.format(epoch))
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
                row = [epoch]
                row += [metrics['map50'], epoch_time]
                row += [train_loss[k] for k in model.subloss_names]
                log_writer.writerow(row)

            # Write to Tensorboard log.
            if self.train_opts.log_tensorboard:
                for key, val in metrics.items():
                    tb_writer.add_scalar(key, val, epoch)
                for key, val in train_loss.items():
                    tb_writer.add_scalar(key, val, epoch)
                for name, param in model.named_parameters():
                    tb_writer.add_histogram(name, param, epoch)

            if (train_uri.startswith('s3://')
                    and ((epoch + 1) % self.train_opts.sync_interval)):
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

            # add one for background class
            num_classes = len(self.task_config.class_map) + 1
            model = MyFasterRCNN(
                self.train_opts.model_arch,
                num_classes,
                self.task_config.chip_size,
                pretrained=False)
            model = model.to(self.device)
            model.load_state_dict(
                torch.load(model_path, map_location=self.device))
            self.model = model

    def predict(self, chips, windows, tmp_dir):
        """Return predictions for a chip using model.

        Args:
            chips: [[height, width, channels], ...] numpy array of chips
            windows: List of boxes that are the windows aligned with the chips.

        Return:
            Labels object containing predictions
        """
        self.load_model(tmp_dir)
        labels = ObjectDetectionLabels.make_empty()
        chips = torch.Tensor(chips).permute((0, 3, 1, 2)) / 255.
        chips = chips.to(self.device)
        model = self.model.eval()

        with torch.no_grad():
            output = model(chips)
            for chip_ind, (chip, window) in enumerate(zip(chips, windows)):
                boxlist = output[chip_ind].cpu()
                boxes = boxlist.boxes.numpy()
                class_ids = boxlist.get_field('labels').numpy()
                scores = boxlist.get_field('scores').numpy()

                boxes = ObjectDetectionLabels.local_to_global(boxes, window)
                labels += ObjectDetectionLabels(
                    boxes, class_ids, scores=scores)

        return labels
