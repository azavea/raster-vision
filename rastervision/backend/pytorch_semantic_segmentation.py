from os.path import join, basename, dirname
import uuid
import zipfile
import glob
from pathlib import Path
import logging
import json
from subprocess import Popen
import os

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastai.vision import (SegmentationItemList, get_transforms, models,
                           unet_learner, Image)
from fastai.callbacks import TrackEpochCallback, OverSamplingCallback
from fastai.basic_train import load_learner
from fastai.vision.transform import dihedral

from rastervision.utils.files import (get_local_path, make_dir, upload_or_copy,
                                      list_paths, download_if_needed,
                                      sync_from_dir, sync_to_dir, str_to_file)
from rastervision.utils.misc import save_img, terminate_at_exit
from rastervision.backend import Backend
from rastervision.data.label import SemanticSegmentationLabels
from rastervision.data.label_source.utils import color_to_triple
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
    if 0 in class_map.get_keys():
        colors = [class_map.get_by_id(i).color for i in range(len(class_map))]
    else:
        # If 0 (ie. the ignore class) is not present in class_map, we need to
        # start indexing class_ids at 1, and insert a color at the beginning to
        # handle NODATA pixels which get mapped to a label class of 0.
        colors = [
            class_map.get_by_id(i).color for i in range(1,
                                                        len(class_map) + 1)
        ]
        colors = ['grey'] + colors
    colors = [color_to_triple(c) for c in colors]
    colors = [tuple([x / 255 for x in c]) for c in colors]
    cmap = matplotlib.colors.ListedColormap(colors)

    def _make_debug_chips(split):
        debug_chips_dir = join(tmp_dir, '{}-debug-chips'.format(split))
        zip_path = join(tmp_dir, '{}-debug-chips.zip'.format(split))
        zip_uri = join(train_uri, '{}-debug-chips.zip'.format(split))
        make_dir(debug_chips_dir)
        dl = data.train_dl if split == 'train' else data.valid_dl
        i = 0
        for _, (x_batch, y_batch) in enumerate(dl):
            for x, y in zip(x_batch, y_batch):
                x = x.squeeze()
                y = y.squeeze()

                # fastai has an x.show(y=y) method, but we need to plot the
                # debug chips ourselves in order to use
                # a custom color map that matches the colors in the class_map.
                # This could be a good things to contribute upstream to fastai.
                plt.axis('off')
                plt.imshow(x.data.permute((1, 2, 0)).detach().cpu().numpy())
                plt.imshow(
                    y.data.squeeze().detach().cpu().numpy(),
                    alpha=0.4,
                    vmin=0,
                    vmax=len(colors),
                    cmap=cmap)
                plt.savefig(
                    join(debug_chips_dir, '{}.png'.format(i)), figsize=(3, 3))
                plt.close()
                i += 1

                if i > max_count:
                    break
            if i > max_count:
                break

        zipdir(debug_chips_dir, zip_path)
        upload_or_copy(zip_path, zip_uri)

    _make_debug_chips('train')
    _make_debug_chips('val')


def get_oversampling_weights(dataset, rare_class_ids, rare_target_prop):
    """Return weight vector for oversampling chips with rare classes.

    Args:
        dataset: PyTorch DataSet with semantic segmentation data
        rare_class_ids: list of rare class ids
        rare_target_prop: desired probability of sampling a chip covering the
            rare classes
    """

    def filter_chip_inds():
        chip_inds = []
        for i, (x, y) in enumerate(dataset):
            match = False
            for class_id in rare_class_ids:
                if torch.any(y.data == class_id):
                    match = True
                    break
            if match:
                chip_inds.append(i)
        return chip_inds

    def get_sample_weights(num_samples, rare_chip_inds, rare_target_prob):
        rare_weight = rare_target_prob / len(rare_chip_inds)
        common_weight = (1 - rare_target_prob) / (
            num_samples - len(rare_chip_inds))
        weights = torch.full((num_samples, ), common_weight)
        weights[rare_chip_inds] = rare_weight
        return weights

    chip_inds = filter_chip_inds()
    print('prop of rare chips before oversampling: ',
          len(chip_inds) / len(dataset))
    weights = get_sample_weights(len(dataset), chip_inds, rare_target_prop)
    return weights


def tta_predict(learner, im_arr):
    """Use test-time augmentation to make predictions for a single image.

    This uses the dihedral transform to make 8 flipped/rotated version of the
    input, makes a prediction for each one, and averages the predictive
    distributions together. This will take 8x the time for a small accuracy
    improvement.

    Args:
        learner: fastai Learner object for semantic segmentation
        im_arr: (Tensor) of shape (nb_channels, height, width)

    Returns:
        (numpy.ndarray) of shape (height, width) containing predicted class ids
    """
    # Note: we are not using the TTA method built into fastai because it only
    # works on image classification problems (and this is undocumented).
    # We should consider contributing this upstream to fastai.
    probs = []
    for k in range(8):
        trans_im = dihedral(Image(im_arr), k)
        o = learner.predict(trans_im)[2]
        # https://forums.fast.ai/t/how-best-to-have-get-preds-or-tta-apply-specified-transforms/40731/9
        o = Image(o)
        if k == 5:
            o = dihedral(o, 6)
        elif k == 6:
            o = dihedral(o, 5)
        else:
            o = dihedral(o, k)
        probs.append(o.data)

    label_arr = torch.stack(probs).mean(0).argmax(0).numpy()
    return label_arr


class PyTorchSemanticSegmentation(Backend):
    """Semantic segmentation backend using PyTorch and fastai."""

    def __init__(self, task_config, backend_opts, train_opts):
        """Constructor.

        Args:
            task_config: (SemanticSegmentationConfig)
            backend_opts: (simple_backend_config.BackendOptions)
            train_opts: (pytorch_semantic_segmentation_backend_config.TrainOptions)
        """
        self.task_config = task_config
        self.backend_opts = backend_opts
        self.train_opts = train_opts
        self.inf_learner = None

        torch_cache_dir = '/opt/data/torch-cache'
        os.environ['TORCH_HOME'] = torch_cache_dir

    def log_options(self):
        log.info('backend_opts:\n' +
                 json.dumps(self.backend_opts.__dict__, indent=4))
        log.info('train_opts:\n' +
                 json.dumps(self.train_opts.__dict__, indent=4))

    def process_scene_data(self, scene, data, tmp_dir):
        """Make training chips for a scene.

        This writes a set of image chips to {scene_id}/img/{scene_id}-{ind}.png
        and corresponding label chips to {scene_id}/labels/{scene_id}-{ind}.png.

        Args:
            scene: (rv.data.Scene)
            data: (rv.data.Dataset)
            tmp_dir: (str) path to temp directory

        Returns:
            (str) path to directory with scene chips {tmp_dir}/{scene_id}
        """
        scene_dir = join(tmp_dir, str(scene.id))
        img_dir = join(scene_dir, 'img')
        labels_dir = join(scene_dir, 'labels')

        make_dir(img_dir)
        make_dir(labels_dir)

        for ind, (chip, window, labels) in enumerate(data):
            chip_path = join(img_dir, '{}-{}.png'.format(scene.id, ind))
            label_path = join(labels_dir, '{}-{}.png'.format(scene.id, ind))

            label_im = labels.get_label_arr(window).astype(np.uint8)
            save_img(label_im, label_path)
            save_img(chip, chip_path)

        return scene_dir

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """Write zip file with chips for a set of scenes.

        This writes a zip file for a group of scenes at {chip_uri}/{uuid}.zip containing:
        train-img/{scene_id}-{ind}.png
        train-labels/{scene_id}-{ind}.png
        val-img/{scene_id}-{ind}.png
        val-labels/{scene_id}-{ind}.png

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

            def _write_zip(results, split):
                for scene_dir in results:
                    scene_paths = glob.glob(join(scene_dir, '**/*.png'))
                    for p in scene_paths:
                        zipf.write(
                            p,
                            join(
                                '{}-{}'.format(split,
                                               dirname(p).split('/')[-1]),
                                basename(p)))

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
        def get_label_path(im_path):
            return Path(str(im_path.parent)[:-4] + '-labels') / im_path.name

        size = self.task_config.chip_size
        class_map = self.task_config.class_map
        classes = class_map.get_class_names()
        if 0 not in class_map.get_keys():
            classes = ['nodata'] + classes
        num_workers = 0 if self.train_opts.debug else 4

        data = SegmentationItemList \
            .from_folder(chip_dir) \
            .split_by_folder(train='train-img', valid='val-img')
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

        data = SegmentationItemList(items, chip_dir) \
            .split_by_folder(train='train-img', valid='val-img') \
            .label_from_func(get_label_path, classes=classes) \
            .transform(get_transforms(flip_vert=self.train_opts.flip_vert),
                       size=size, tfm_y=True) \
            .databunch(bs=self.train_opts.batch_size, num_workers=num_workers)
        print(data)

        # Setup learner.
        ignore_idx = 0
        metrics = [
            Precision(average='weighted', clas_idx=1, ignore_idx=ignore_idx),
            Recall(average='weighted', clas_idx=1, ignore_idx=ignore_idx),
            FBeta(
                average='weighted', clas_idx=1, beta=1, ignore_idx=ignore_idx)
        ]
        model_arch = getattr(models, self.train_opts.model_arch)
        learn = unet_learner(
            data,
            model_arch,
            metrics=metrics,
            wd=self.train_opts.weight_decay,
            bottle=True,
            path=train_dir)
        learn.unfreeze()

        if self.train_opts.mixed_prec and torch.cuda.is_available():
            # This loss_scale works for Resnet 34 and 50. You might need to adjust this
            # for other models.
            learn = learn.to_fp16(loss_scale=256)

        # Setup callbacks and train model.
        model_path = get_local_path(self.backend_opts.model_uri, tmp_dir)

        pretrained_uri = self.backend_opts.pretrained_uri
        if pretrained_uri:
            print('Loading weights from pretrained_uri: {}'.format(
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

        oversample = self.train_opts.oversample
        if oversample:
            weights = get_oversampling_weights(data.train_ds,
                                               oversample['rare_class_ids'],
                                               oversample['rare_target_prop'])
            oversample_callback = OverSamplingCallback(learn, weights=weights)
            callbacks.append(oversample_callback)

        if self.train_opts.debug:
            if oversample:
                oversample_callback.on_train_begin()
            make_debug_chips(data, class_map, tmp_dir, train_uri)

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
                print('lr_find() found lr: {}'.format(lr))
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

    def predict(self, chips, windows, tmp_dir):
        """Return a prediction for a single chip.

        Args:
            chips: (numpy.ndarray) of shape (1, height, width, nb_channels)
                containing a single imagery chip
            windows: List containing a single (Box) window which is aligned
                with the chip

        Return:
            (SemanticSegmentationLabels) containing predictions
        """
        self.load_model(tmp_dir)

        chip = torch.Tensor(chips[0]).permute((2, 0, 1)) / 255.
        im = Image(chip)
        self.inf_learner.data.single_ds.tfmargs[
            'size'] = self.task_config.predict_chip_size
        self.inf_learner.data.single_ds.tfmargs_y[
            'size'] = self.task_config.predict_chip_size

        if self.train_opts.tta:
            label_arr = tta_predict(self.inf_learner, chip)
        else:
            label_arr = self.inf_learner.predict(im)[1].squeeze().numpy()

        # TODO better explanation
        # Return "trivial" instance of SemanticSegmentationLabels that holds a single
        # window and has ability to get labels for that one window.
        def label_fn(_window):
            if _window == windows[0]:
                return label_arr
            else:
                raise ValueError('Trying to get labels for unknown window.')

        return SemanticSegmentationLabels(windows, label_fn)
