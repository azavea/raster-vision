from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List,
                    Optional, Tuple, Union, Type)
from typing_extensions import Literal
from abc import ABC, abstractmethod
from os.path import join, isfile, basename, isdir
import csv
import warnings
import time
import datetime
import shutil
import logging
from subprocess import Popen
import numbers
from pprint import pformat

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from rastervision.pipeline.file_system import (
    sync_to_dir, json_to_file, file_to_json, make_dir, zipdir,
    download_if_needed, download_or_copy, sync_from_dir, get_local_path, unzip,
    str_to_file, is_local, get_tmp_dir)
from rastervision.pipeline.file_system.utils import file_exists
from rastervision.pipeline.utils import terminate_at_exit
from rastervision.pipeline.config import (build_config, upgrade_config,
                                          save_pipeline_config)
from rastervision.pytorch_learner.utils import (get_hubconf_dir_from_cfg)
from rastervision.pytorch_learner.dataset.visualizer import Visualizer

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.utils.data import Dataset, Sampler

    from rastervision.pytorch_learner import (LearnerConfig,
                                              LearnerPipelineConfig)

warnings.filterwarnings('ignore')

MODULES_DIRNAME = 'modules'
TRANSFORMS_DIRNAME = 'custom_albumentations_transforms'

log = logging.getLogger(__name__)

MetricDict = Dict[str, float]


def log_system_details():
    """Log some system details."""
    import os
    import sys
    import psutil
    # CPUs
    log.info(f'Physical CPUs: {psutil.cpu_count(logical=False)}')
    log.info(f'Logical CPUs: {psutil.cpu_count(logical=True)}')
    # memory usage
    mem_stats = psutil.virtual_memory()._asdict()
    log.info(f'Total memory: {mem_stats["total"] / 2**30: .2f} GB')

    # disk usage
    if os.path.isdir('/opt/data/'):
        disk_stats = psutil.disk_usage('/opt/data')._asdict()
        log.info(
            f'Size of /opt/data volume: {disk_stats["total"] / 2**30: .2f} GB')
    disk_stats = psutil.disk_usage('/')._asdict()
    log.info(f'Size of / volume: {disk_stats["total"] / 2**30: .2f} GB')

    # python
    log.info(f'Python version: {sys.version}')
    # nvidia GPU
    try:
        with os.popen('nvcc --version') as f:
            log.info(f.read())
        with os.popen('nvidia-smi') as f:
            log.info(f.read())
        log.info('Devices:')
        device_query = ' '.join([
            'nvidia-smi', '--format=csv',
            '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free'
        ])
        with os.popen(device_query) as f:
            log.info(f.read())
    except FileNotFoundError:
        pass
    # pytorch and CUDA
    log.info(f'PyTorch version: {torch.__version__}')
    log.info(f'CUDA available: {torch.cuda.is_available()}')
    log.info(f'CUDA version: {torch.version.cuda}')
    log.info(f'CUDNN version: {torch.backends.cudnn.version()}')
    log.info(f'Number of CUDA devices: {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        log.info(f'Active CUDA Device: GPU {torch.cuda.current_device()}')


class Learner(ABC):
    """Abstract training and prediction routines for a model.

    This can be subclassed to handle different computer vision tasks.

    The datasets, model, optimizer, and schedulers will be generated from the
    cfg if not specified in the constructor.

    If instantiated with training=False, the training apparatus (loss,
    optimizer, scheduler, logging, etc.) will not be set up and the model will
    be put into eval mode.

    Note that various training and prediction methods have the side effect of
    putting Learner.model into training or eval mode. No attempt is made to put the
    model back into the mode it was previously in.
    """

    def __init__(self,
                 cfg: 'LearnerConfig',
                 output_dir: Optional[str] = None,
                 train_ds: Optional['Dataset'] = None,
                 valid_ds: Optional['Dataset'] = None,
                 test_ds: Optional['Dataset'] = None,
                 model: Optional[nn.Module] = None,
                 loss: Optional[Callable] = None,
                 optimizer: Optional['Optimizer'] = None,
                 epoch_scheduler: Optional['_LRScheduler'] = None,
                 step_scheduler: Optional['_LRScheduler'] = None,
                 tmp_dir: Optional[str] = None,
                 model_weights_path: Optional[str] = None,
                 model_def_path: Optional[str] = None,
                 loss_def_path: Optional[str] = None,
                 training: bool = True):
        """Constructor.

        Args:
            cfg (LearnerConfig): LearnerConfig.
            train_ds (Optional[Dataset], optional): The dataset to use for
                training. If None, will be generated from cfg.data.
                Defaults to None.
            valid_ds (Optional[Dataset], optional): The dataset to use for
                validation. If None, will be generated from cfg.data.
                Defaults to None.
            test_ds (Optional[Dataset], optional): The dataset to use for
                testing. If None, will be generated from cfg.data.
                Defaults to None.
            model (Optional[nn.Module], optional): The model. If None,
                will be generated from cfg.model. Defaults to None.
            loss (Optional[Callable], optional): The loss function.
                If None, will be generated from cfg.solver.
                Defaults to None.
            optimizer (Optional[Optimizer], optional): The optimizer.
                If None, will be generated from cfg.solver.
                Defaults to None.
            epoch_scheduler (Optional[_LRScheduler], optional): The scheduler
                that updates after each epoch. If None, will be generated from
                cfg.solver. Defaults to None.
            step_scheduler (Optional[_LRScheduler], optional): The scheduler
                that updates after each optimizer-step. If None, will be
                generated from cfg.solver. Defaults to None.
            tmp_dir (Optional[str], optional): A temporary directory to use for
                downloads etc. If None, will be auto-generated.
                Defaults to None.
            model_weights_path (Optional[str], optional): URI of model weights
                to initialize the model with. Defaults to None.
            model_def_path (Optional[str], optional): A local path to a
                directory with a hubconf.py. If provided, the model definition
                is imported from here. This is used when loading an external
                model from a model-bundle. Defaults to None.
            loss_def_path (Optional[str], optional): A local path to a
                directory with a hubconf.py. If provided, the loss function
                definition is imported from here. This is used when loading an
                external loss function from a model-bundle. Defaults to None.
            training (bool, optional): If False, the training apparatus (loss,
                optimizer, scheduler, logging, etc.) will not be set up and the
                model will be put into eval mode. If True, the training
                apparatus will be set up and the model will be put into
                training mode. Defaults to True.
        """
        self.cfg = cfg

        if model is None and cfg.model is None:
            raise ValueError(
                'cfg.model can only be None if a custom model is specified.')

        if tmp_dir is None:
            self._tmp_dir = get_tmp_dir()
            tmp_dir = self._tmp_dir.name
        self.tmp_dir = tmp_dir
        self.device = torch.device('cuda'
                                   if torch.cuda.is_available() else 'cpu')

        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

        self.model = model
        self.loss = loss
        self.opt = optimizer
        self.epoch_scheduler = epoch_scheduler
        self.step_scheduler = step_scheduler

        # ---------------------------
        # Set URIs
        # ---------------------------
        if output_dir is None and cfg.output_uri is None:
            raise ValueError('output_dir or LearnerConfig.output_uri must '
                             'be specified.')
        if output_dir is not None and cfg.output_uri is not None:
            log.warn('Both output_dir and LearnerConfig.output_uri specified. '
                     'LearnerConfig.output_uri will be ignored.')
        if output_dir is None:
            assert cfg.output_uri is not None
            self.output_dir = cfg.output_uri
            self.model_bundle_uri = cfg.get_model_bundle_uri()
        else:
            self.output_dir = output_dir
            self.model_bundle_uri = join(self.output_dir, 'model-bundle.zip')

        if is_local(self.output_dir):
            self.output_dir_local = self.output_dir
            make_dir(self.output_dir_local)
        else:
            self.output_dir_local = get_local_path(self.output_dir, tmp_dir)
            make_dir(self.output_dir_local, force_empty=True)
            if training and not cfg.overfit_mode:
                self.sync_from_cloud()
            log.info(f'Local output dir: {self.output_dir_local}')
            log.info(f'Remote output dir: {self.output_dir}')

        self.modules_dir = join(self.output_dir, MODULES_DIRNAME)
        # ---------------------------

        self.setup_model(
            model_weights_path=model_weights_path,
            model_def_path=model_def_path)

        if training:
            self.setup_training(loss_def_path=loss_def_path)
            self.model.train()
        else:
            self.model.eval()

        self.visualizer = self.get_visualizer_class()(
            cfg.data.class_names, cfg.data.class_colors,
            cfg.data.plot_options.transform,
            cfg.data.plot_options.channel_display_groups)

    def main(self):
        """Main training sequence.

        This plots the dataset, runs a training and validation loop (which will resume if
        interrupted), logs stats, plots predictions, and syncs results to the cloud.
        """
        log_system_details()
        log.info(self.cfg)
        log.info(f'Using device: {self.device}')
        self.log_data_stats()
        self.run_tensorboard()

        cfg = self.cfg
        if not cfg.predict_mode:
            self.plot_dataloaders(self.cfg.data.preview_batch_limit)
            if cfg.overfit_mode:
                self.overfit()
            else:
                self.train()
                if cfg.save_model_bundle:
                    self.save_model_bundle()
        else:
            self.load_checkpoint()

        self.stop_tensorboard()
        if cfg.eval_train:
            self.eval_model('train')
        self.eval_model('valid')
        self.sync_to_cloud()

    def setup_training(self, loss_def_path: Optional[str] = None) -> None:
        cfg = self.cfg

        self.config_path = join(self.output_dir, 'learner-config.json')
        str_to_file(cfg.json(), self.config_path)

        self.log_path = join(self.output_dir_local, 'log.csv')
        self.metric_names = self.build_metric_names()

        # data
        self.setup_data()

        # model
        self.last_model_weights_path = join(self.output_dir_local,
                                            'last-model.pth')
        self.load_checkpoint()

        # optimization
        start_epoch = self.get_start_epoch()
        self.setup_loss(loss_def_path=loss_def_path)
        if self.opt is None:
            self.opt = self.build_optimizer()
        if self.step_scheduler is None:
            self.step_scheduler = self.build_step_scheduler(start_epoch)
        if self.epoch_scheduler is None:
            self.epoch_scheduler = self.build_epoch_scheduler(start_epoch)

        self.setup_tensorboard()

    def sync_to_cloud(self):
        """Sync any output to the cloud at output_uri."""
        sync_to_dir(self.output_dir_local, self.output_dir)

    def sync_from_cloud(self):
        """Sync any previous output in the cloud to output_dir."""
        sync_from_dir(self.output_dir, self.output_dir_local)

    def setup_tensorboard(self):
        """Setup for logging stats to TB."""
        self.tb_writer = None
        if self.cfg.log_tensorboard:
            self.tb_log_dir = join(self.output_dir_local, 'tb-logs')
            make_dir(self.tb_log_dir)
            self.tb_writer = SummaryWriter(log_dir=self.tb_log_dir)

    def run_tensorboard(self):
        """Run TB server serving logged stats."""
        if self.cfg.run_tensorboard:
            log.info('Starting tensorboard process')
            self.tb_process = Popen([
                'tensorboard', '--bind_all',
                '--logdir={}'.format(self.tb_log_dir)
            ])
            terminate_at_exit(self.tb_process)

    def stop_tensorboard(self):
        """Stop TB logging and server if it's running."""
        if self.cfg.log_tensorboard:
            self.tb_writer.close()
            if self.cfg.run_tensorboard:
                self.tb_process.terminate()

    def setup_model(self,
                    model_weights_path: Optional[str] = None,
                    model_def_path: Optional[str] = None) -> None:
        """Setup self.model.

        Args:
            model_weights_path (Optional[str], optional): Path to model
                weights. Will be available when loading from a bundle.
                Defaults to None.
            model_def_path (Optional[str], optional): Path to model definition.
                Will be available when loading from a bundle. Defaults to None.
        """
        if self.model is None:
            self.model = self.build_model(model_def_path=model_def_path)
        self.model.to(device=self.device)
        self.load_init_weights(model_weights_path=model_weights_path)

    def build_model(self, model_def_path: Optional[str] = None) -> nn.Module:
        """Build a PyTorch model."""
        cfg = self.cfg

        in_channels = cfg.data.img_channels
        if in_channels is None:
            log.warn('DataConfig.img_channels is None. Defaulting to 3.')
            in_channels = 3

        model = cfg.model.build(
            num_classes=cfg.data.num_classes,
            in_channels=in_channels,
            save_dir=self.modules_dir,
            hubconf_dir=model_def_path)
        return model

    def setup_loss(self, loss_def_path: Optional[str] = None) -> None:
        """Setup self.loss.

        Args:
            loss_def_path (str, optional): Loss definition path. Will be
            available when loading from a bundle. Defaults to None.
        """
        if self.loss is None:
            self.loss = self.build_loss(loss_def_path=loss_def_path)

        if self.loss is not None and isinstance(self.loss, nn.Module):
            self.loss.to(self.device)

    def build_loss(self, loss_def_path: Optional[str] = None) -> Callable:
        """Build a loss Callable."""
        cfg = self.cfg
        loss = cfg.solver.build_loss(
            num_classes=cfg.data.num_classes,
            save_dir=self.modules_dir,
            hubconf_dir=loss_def_path)
        return loss

    def get_collate_fn(self) -> Optional[callable]:
        """Returns a custom collate_fn to use in DataLoader.

        None is returned if default collate_fn should be used.

        See https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        """
        return None

    def get_train_sampler(self, train_ds: 'Dataset') -> Optional['Sampler']:
        """Return a sampler to use for the training dataloader or None to not use any."""
        return None

    def setup_data(self):
        """Set datasets and dataLoaders for train, validation, and test sets.
        """
        if self.train_ds is None or self.valid_ds is None:
            train_ds, valid_ds, test_ds = self.build_datasets()
            if self.train_ds is None:
                self.train_ds = train_ds
            if self.valid_ds is None:
                self.valid_ds = valid_ds
            if self.test_ds is None:
                self.test_ds = test_ds
        self.train_dl, self.valid_dl, self.test_dl = self.build_dataloaders()

    def build_datasets(self) -> Tuple['Dataset', 'Dataset', 'Dataset']:
        log.info(f'Building datasets ...')
        cfg = self.cfg
        train_ds, val_ds, test_ds = self.cfg.data.build(
            tmp_dir=self.tmp_dir,
            overfit_mode=cfg.overfit_mode,
            test_mode=cfg.test_mode)
        return train_ds, val_ds, test_ds

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Set the DataLoaders for train, validation, and test sets."""

        batch_sz = self.cfg.solver.batch_sz
        num_workers = self.cfg.data.num_workers
        collate_fn = self.get_collate_fn()

        train_sampler = self.get_train_sampler(self.train_ds)
        train_shuffle = train_sampler is None
        # batchnorm layers expect batch size > 1 during training
        train_drop_last = (len(self.train_ds) % batch_sz) == 1
        train_dl = DataLoader(
            self.train_ds,
            batch_size=batch_sz,
            shuffle=train_shuffle,
            drop_last=train_drop_last,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler)

        val_dl = DataLoader(
            self.valid_ds,
            batch_size=batch_sz,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True)

        test_dl = None
        if self.test_ds is not None and len(self.test_ds) > 0:
            test_dl = DataLoader(
                self.test_ds,
                batch_size=batch_sz,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True)

        return train_dl, val_dl, test_dl

    def log_data_stats(self):
        """Log stats about each DataSet."""
        if self.train_ds is not None:
            log.info(f'train_ds: {len(self.train_ds)} items')
        if self.valid_ds is not None:
            log.info(f'valid_ds: {len(self.valid_ds)} items')
        if self.test_ds is not None:
            log.info(f'test_ds: {len(self.test_ds)} items')

    def build_optimizer(self) -> 'Optimizer':
        """Returns optimizer."""
        return self.cfg.solver.build_optimizer(self.model)

    def build_step_scheduler(self, start_epoch: int = 0) -> '_LRScheduler':
        """Returns an LR scheduler that changes the LR each step."""
        return self.cfg.solver.build_step_scheduler(
            optimizer=self.opt,
            train_ds_sz=len(self.train_ds),
            last_epoch=(start_epoch - 1))

    def build_epoch_scheduler(self, start_epoch: int = 0) -> '_LRScheduler':
        """Returns an LR scheduler that changes the LR each epoch."""
        return self.cfg.solver.build_epoch_scheduler(
            optimizer=self.opt, last_epoch=(start_epoch - 1))

    def build_metric_names(self) -> List[str]:
        """Returns names of metrics used to validate model at each epoch."""
        metric_names = [
            'epoch', 'train_time', 'valid_time', 'train_loss', 'val_loss',
            'avg_f1', 'avg_precision', 'avg_recall'
        ]

        for label in self.cfg.data.class_names:
            metric_names.extend([
                '{}_f1'.format(label), '{}_precision'.format(label),
                '{}_recall'.format(label)
            ])
        return metric_names

    @abstractmethod
    def get_visualizer_class(self) -> Type[Visualizer]:
        """Returns a Visualizer class object for plotting data samples."""
        pass

    @abstractmethod
    def train_step(self, batch: Any, batch_ind: int) -> MetricDict:
        """Compute loss for a single training batch.

        Args:
            batch: batch data needed to compute loss
            batch_ind: index of batch within epoch

        Returns:
            dict with 'train_loss' as key and possibly other losses
        """
        pass

    @abstractmethod
    def validate_step(self, batch: Any, batch_ind: int) -> MetricDict:
        """Compute metrics on validation batch.

        Args:
            batch: batch data needed to compute validation metrics
            batch_ind: index of batch within epoch

        Returns:
            dict with metric names mapped to metric values
        """
        pass

    def train_end(self, outputs: List[MetricDict],
                  num_samples: int) -> MetricDict:
        """Aggregate the ouput of train_step at the end of the epoch.

        Args:
            outputs: a list of outputs of train_step
            num_samples: total number of training samples processed in epoch
        """
        metrics = {}
        for k in outputs[0].keys():
            metrics[k] = torch.stack([o[k] for o in outputs
                                      ]).sum().item() / num_samples
        return metrics

    def validate_end(self, outputs: List[MetricDict],
                     num_samples: int) -> MetricDict:
        """Aggregate the ouput of validate_step at the end of the epoch.

        Args:
            outputs: a list of outputs of validate_step
            num_samples: total number of validation samples processed in epoch
        """
        metrics = {}
        for k in outputs[0].keys():
            metrics[k] = torch.stack([o[k] for o in outputs
                                      ]).sum().item() / num_samples
        return metrics

    def post_forward(self, x: Any) -> Any:
        """Post process output of call to model().

        Useful for when predictions are inside a structure returned by model().
        """
        return x

    def prob_to_pred(self, x: Tensor) -> Tensor:
        """Convert a Tensor with prediction probabilities to class ids.

        The class ids should be the classes with the maximum probability.
        """
        raise NotImplementedError()

    def to_batch(self, x: Tensor) -> Tensor:
        """Ensure that image array has batch dimension.

        Args:
            x: assumed to be either image or batch of images

        Returns:
            x with extra batch dimension of length 1 if needed
        """
        if x.ndim == 3:
            x = x[None, ...]
        return x

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize x to [0, 1].

        If x.dtype is a subtype of np.unsignedinteger, normalize it to
        [0, 1] using the max possible value of that dtype. Otherwise, assume
        it is in [0, 1] already and do nothing.

        Args:
            x (np.ndarray): an image or batch of images
        Returns:
            the same array scaled to [0, 1].
        """
        if np.issubdtype(x.dtype, np.unsignedinteger):
            max_val = np.iinfo(x.dtype).max
            x = x.astype(float) / max_val
        return x

    def predict(self, x: Tensor, raw_out: bool = False) -> Any:
        """Make prediction for an image or batch of images.

        Args:
            x (Tensor): Image or batch of images as a float Tensor with pixel
                values normalized to [0, 1].
            raw_out (bool): if True, return prediction probabilities

        Returns:
            the predictions, in probability form if raw_out is True, in class_id form
                otherwise
        """
        x = self.to_batch(x).float()
        x = self.to_device(x, self.device)
        with torch.inference_mode():
            out = self.model(x)
            if not raw_out:
                out = self.prob_to_pred(self.post_forward(out))
        out = self.to_device(out, 'cpu')
        return out

    def output_to_numpy(self, out: Tensor) -> np.ndarray:
        """Convert output of model to numpy format.

        Args:
            out: the output of the model in PyTorch format

        Returns: the output of the model in numpy format
        """
        return out.numpy()

    def numpy_predict(self, x: np.ndarray,
                      raw_out: bool = False) -> np.ndarray:
        """Make a prediction using an image or batch of images in numpy format.
        If x.dtype is a subtype of np.unsignedinteger, it will be normalized
        to [0, 1] using the max possible value of that dtype. Otherwise, x will
        be assumed to be in [0, 1] already and will be cast to torch.float32
        directly.

        Args:
            x: (ndarray) of shape [height, width, channels] or
                [batch_sz, height, width, channels]
            raw_out: if True, return prediction probabilities

        Returns:
            predictions using numpy arrays
        """
        transform, _ = self.cfg.data.get_data_transforms()
        x = self.normalize_input(x)
        x = self.to_batch(x)
        x = np.stack([transform(image=img)['image'] for img in x])
        x = torch.from_numpy(x)
        x = x.permute((0, 3, 1, 2))
        out = self.predict(x, raw_out=raw_out)
        return self.output_to_numpy(out)

    def predict_dataset(self,
                        dataset: 'Dataset',
                        return_format: Literal['xyz', 'yz', 'z'] = 'z',
                        raw_out: bool = True,
                        numpy_out: bool = False,
                        predict_kw: dict = {},
                        dataloader_kw: dict = {},
                        progress_bar: bool = True,
                        progress_bar_kw: dict = {}
                        ) -> Union[Iterator[Any], Iterator[Tuple[Any, ...]]]:
        """Returns an iterator over predictions on the given dataset.

        Args:
            dataset (Dataset): The dataset to make predictions on.
            return_format (Literal['xyz', 'yz', 'z'], optional): Format of the
                return elements of the returned iterator. Must be one of:
                'xyz', 'yz', and 'z'. If 'xyz', elements are 3-tuples of x, y,
                and z. If 'yz', elements are 2-tuples of y and z. If 'z',
                elements are (non-tuple) values of z. Where x = input image,
                y = ground truth, and z = prediction. Defaults to 'z'.
            raw_out (bool, optional): If true, return raw predicted scores.
                Defaults to True.
            numpy_out (bool, optional): If True, convert predictions to numpy
                arrays before returning. Defaults to False.
            predict_kw (dict): Dict with keywords passed to Learner.predict().
                Useful if a Learner subclass implements a custom predict()
                method.
            dataloader_kw (dict): Dict with keywords passed to the DataLoader
                constructor.
            progress_bar (bool, optional): If True, display a progress bar.
                Since this function returns an iterator, the progress bar won't
                be visible until the iterator is consumed. Defaults to True.
            progress_bar_kw (dict): Dict with keywords passed to tqdm.

        Raises:
            ValueError: If return_format is not one of the allowed values.

        Returns:
            Union[Iterator[Any], Iterator[Tuple[Any, ...]]]: If return_format
                is 'z', the returned value is an iterator of whatever type the
                predictions are. Otherwise, the returned value is an iterator
                of tuples.
        """

        if return_format not in {'xyz', 'yz', 'z'}:
            raise ValueError('return_format must be one of "xyz", "yz", "z".')

        dl_kw = dict(
            collate_fn=self.get_collate_fn(),
            batch_size=self.cfg.solver.batch_sz,
            num_workers=self.cfg.data.num_workers,
            shuffle=False,
            pin_memory=True)
        dl_kw.update(dataloader_kw)
        dl = DataLoader(dataset, **dl_kw)

        preds = self.predict_dataloader(
            dl,
            return_format=return_format,
            raw_out=raw_out,
            batched_output=False,
            predict_kw=predict_kw)

        if numpy_out:
            if return_format == 'z':
                preds = (self.output_to_numpy(p) for p in preds)
            else:
                # only convert z
                preds = ((*p[:-1], self.output_to_numpy(p[-1])) for p in preds)

        if progress_bar:
            pb_kw = dict(desc='Predicting', total=len(dataset))
            pb_kw.update(progress_bar_kw)
            preds = tqdm(preds, **pb_kw)

        return preds

    def predict_dataloader(
            self,
            dl: DataLoader,
            batched_output: bool = True,
            return_format: Literal['xyz', 'yz', 'z'] = 'z',
            raw_out: bool = True,
            predict_kw: dict = {}
    ) -> Union[Iterator[Any], Iterator[Tuple[Any, ...]]]:
        """Returns an iterator over predictions on the given dataloader.

        Args:
            dl (DataLoader): The dataloader to make predictions on.
            batched_output (bool, optional): If True, return batches of
                x, y, z as defined by the dataloader. If False, unroll the
                batches into individual items. Defaults to True.
            return_format (Literal['xyz', 'yz', 'z'], optional): Format of the
                return elements of the returned iterator. Must be one of:
                'xyz', 'yz', and 'z'. If 'xyz', elements are 3-tuples of x, y,
                and z. If 'yz', elements are 2-tuples of y and z. If 'z',
                elements are (non-tuple) values of z. Where x = input image,
                y = ground truth, and z = prediction. Defaults to 'z'.
            raw_out (bool, optional): If true, return raw predicted scores.
                Defaults to True.
            predict_kw (dict): Dict with keywords passed to Learner.predict().
                Useful if a Learner subclass implements a custom predict()
                method.

        Raises:
            ValueError: If return_format is not one of the allowed values.

        Returns:
            Union[Iterator[Any], Iterator[Tuple[Any, ...]]]: If return_format
                is 'z', the returned value is an iterator of whatever type the
                predictions are. Otherwise, the returned value is an iterator
                of tuples.
        """

        if return_format not in {'xyz', 'yz', 'z'}:
            raise ValueError('return_format must be one of "xyz", "yz", "z".')

        preds = self._predict_dataloader(
            dl,
            raw_out=raw_out,
            batched_output=batched_output,
            predict_kw=predict_kw)

        if return_format == 'yz':
            preds = ((y, z) for _, y, z in preds)
        elif return_format == 'z':
            preds = (z for _, _, z in preds)

        return preds

    def _predict_dataloader(
            self,
            dl: DataLoader,
            raw_out: bool = True,
            batched_output: bool = True,
            predict_kw: dict = {}) -> Iterator[Tuple[Tensor, Any, Any]]:
        """Returns an iterator over predictions on the given dataloader.

        Args:
            dl (DataLoader): The dataloader to make predictions on.
            batched_output (bool, optional): If True, return batches of
                x, y, z as defined by the dataloader. If False, unroll the
                batches into individual items. Defaults to True.
            raw_out (bool, optional): If true, return raw predicted scores.
                Defaults to True.
            predict_kw (dict): Dict with keywords passed to Learner.predict().
                Useful if a Learner subclass implements a custom predict()
                method.

        Raises:
            ValueError: If return_format is not one of the allowed values.

        Yields:
            Iterator[Tuple[Tensor, Any, Any]]: 3-tuples of x, y, and z, which
                might or might not be batched depending on the batched_output
                argument.
        """
        self.model.eval()

        for x, y in dl:
            x = self.to_device(x, self.device)
            z = self.predict(x, raw_out=raw_out, **predict_kw)
            x = self.to_device(x, 'cpu')
            y = self.to_device(y, 'cpu') if y is not None else y
            z = self.to_device(z, 'cpu')
            if batched_output:
                yield x, y, z
            else:
                for _x, _y, _z in zip(x, y, z):
                    yield _x, _y, _z

    def get_dataloader(self, split: str) -> DataLoader:
        """Get the DataLoader for a split.

        Args:
            split: a split name which can be train, valid, or test
        """
        if split == 'train':
            return self.train_dl
        elif split == 'valid':
            return self.valid_dl
        elif split == 'test':
            return self.test_dl
        else:
            raise ValueError('{} is not a valid split'.format(split))

    def plot_predictions(self,
                         split: str,
                         batch_limit: Optional[int] = None,
                         show: bool = False):
        """Plot predictions for a split.

        Uses the first batch for the corresponding DataLoader.

        Args:
            split: dataset split. Can be train, valid, or test.
            batch_limit: optional limit on (rendered) batch size
        """
        log.info(
            f'Making and plotting sample predictions on the {split} set...')
        dl = self.get_dataloader(split)
        output_path = join(self.output_dir_local, f'{split}_preds.png')
        preds = self.predict_dataloader(
            dl, return_format='xyz', batched_output=True, raw_out=True)
        x, y, z = next(preds)
        self.visualizer.plot_batch(
            x, y, output_path, z=z, batch_limit=batch_limit, show=show)
        log.info(f'Sample predictions written to {output_path}.')

    def plot_dataloader(self,
                        dl: DataLoader,
                        output_path: str,
                        batch_limit: Optional[int] = None,
                        show: bool = False):
        """Plot images and ground truth labels for a DataLoader."""
        x, y = next(iter(dl))
        self.visualizer.plot_batch(
            x, y, output_path, batch_limit=batch_limit, show=show)

    def plot_dataloaders(self,
                         batch_limit: Optional[int] = None,
                         show: bool = False):
        """Plot images and ground truth labels for all DataLoaders."""
        if self.train_dl:
            log.info('Plotting sample training batch.')
            self.plot_dataloader(
                self.train_dl,
                output_path=join(self.output_dir_local,
                                 'dataloaders/train.png'),
                batch_limit=batch_limit,
                show=show)
        if self.valid_dl:
            log.info('Plotting sample validation batch.')
            self.plot_dataloader(
                self.valid_dl,
                output_path=join(self.output_dir_local,
                                 'dataloaders/valid.png'),
                batch_limit=batch_limit,
                show=show)
        if self.test_dl:
            log.info('Plotting sample test batch.')
            self.plot_dataloader(
                self.test_dl,
                output_path=join(self.output_dir_local,
                                 'dataloaders/test.png'),
                batch_limit=batch_limit,
                show=show)

    @classmethod
    def from_model_bundle(cls: Type,
                          model_bundle_uri: str,
                          tmp_dir: Optional[str] = None,
                          cfg: Optional['LearnerConfig'] = None,
                          training: bool = False,
                          **kwargs) -> 'Learner':
        """Create a Learner from a model bundle.

        .. note::

            This is the bundle saved in ``train/model-bundle.zip`` and not
            ``bundle/model-bundle.zip``.

        Args:
            model_bundle_uri (str): URI of the model bundle.
            tmp_dir (Optional[str], optional): Optional temporary directory.
                Will be used for unzipping bundle and also passed to the
                default constructor. If None, will be auto-generated.
                Defaults to None.
            cfg (Optional[LearnerConfig], optional): If None, will be read from
                the bundle. Defaults to None.
            training (bool, optional): If False, the training apparatus (loss,
                optimizer, scheduler, logging, etc.) will not be set up and the
                model will be put into eval mode. If True, the training
                apparatus will be set up and the model will be put into
                training mode. Defaults to True.
            **kwargs: See :meth:`.Learner.__init__`.

        Raises:
            FileNotFoundError: If using custom Albumentations transforms and
                definition file is not found in bundle.

        Returns:
            Learner: Object of the Learner subclass on which this was called.
        """
        log.info(f'Loading learner from bundle {model_bundle_uri}.')
        if tmp_dir is None:
            _tmp_dir = get_tmp_dir()
            tmp_dir = _tmp_dir.name
        model_bundle_path = download_if_needed(model_bundle_uri)
        model_bundle_dir = join(tmp_dir, 'model-bundle')
        log.info(f'Unzipping model-bundle to {model_bundle_dir}')
        unzip(model_bundle_path, model_bundle_dir)

        model_weights_path = join(model_bundle_dir, 'model.pth')

        if cfg is None:
            config_path = join(model_bundle_dir, 'pipeline-config.json')

            config_dict = file_to_json(config_path)
            config_dict = upgrade_config(config_dict)

            learner_pipeline_cfg: 'LearnerPipelineConfig' = build_config(
                config_dict)
            cfg = learner_pipeline_cfg.learner

        hub_dir = join(model_bundle_dir, MODULES_DIRNAME)
        model_def_path = None
        loss_def_path = None

        # retrieve existing model definition, if available
        ext_cfg = cfg.model.external_def if cfg.model is not None else None
        if ext_cfg is not None:
            model_def_path = get_hubconf_dir_from_cfg(ext_cfg, parent=hub_dir)
            log.info(
                f'Using model definition found in bundle: {model_def_path}')

        # retrieve existing loss function definition, if available
        ext_cfg = cfg.solver.external_loss_def
        if ext_cfg is not None and training:
            loss_def_path = get_hubconf_dir_from_cfg(ext_cfg, parent=hub_dir)
            log.info(f'Using loss definition found in bundle: {loss_def_path}')

        # use the definition file(s) saved in the bundle
        custom_transforms = cfg.data.get_custom_albumentations_transforms()
        if len(custom_transforms) > 0:
            for tf in custom_transforms:
                # convert the relative path to a full path
                tf_bundle_path = join(tmp_dir, tf['lambda_transforms_path'])
                tf['lambda_transforms_path'] = tf_bundle_path
                if not file_exists(tf['lambda_transforms_path']):
                    raise FileNotFoundError(
                        f'Custom transform definition file {tf_bundle_path} '
                        'was not found inside the bundle.')
            # config has been altered, so re-validate
            cfg = build_config(cfg.dict())

        if cfg.model is None and kwargs.get('model') is None:
            raise ValueError(
                'Model definition is not saved in the model-bundle. '
                'Please specify the model explicitly.')

        if cls == Learner:
            if len(kwargs) > 0:
                raise ValueError('kwargs are only supported if calling '
                                 '.from_model_bundle() on a Learner subclass '
                                 '-- not Learner itself.')
            learner: cls = cfg.build(
                tmp_dir=tmp_dir,
                model_weights_path=model_weights_path,
                model_def_path=model_def_path,
                loss_def_path=loss_def_path,
                training=training)
        else:
            learner = cls(
                cfg=cfg,
                tmp_dir=tmp_dir,
                model_weights_path=model_weights_path,
                model_def_path=model_def_path,
                loss_def_path=loss_def_path,
                training=training,
                **kwargs)
        return learner

    def save_model_bundle(self):
        """Save a model bundle.

        This is a zip file with the model weights in .pth format and a serialized
        copy of the LearningConfig, which allows for making predictions in the future.
        """
        from rastervision.pytorch_learner.learner_pipeline_config import (
            LearnerPipelineConfig)

        if self.cfg.model is None:
            log.warning(
                'Model was not configured via ModelConfig, and therefore, '
                'will not be reconstructable form the model-bundle. You will '
                'need to initialize the model yourself and pass it to '
                'from_model_bundle().')

        log.info('Creating bundle.')
        model_bundle_dir = join(self.tmp_dir, 'model-bundle')
        make_dir(model_bundle_dir, force_empty=True)

        self._bundle_model(model_bundle_dir)
        self._bundle_modules(model_bundle_dir)
        self._bundle_transforms(model_bundle_dir)

        pipeline_cfg = LearnerPipelineConfig(learner=self.cfg)
        save_pipeline_config(pipeline_cfg,
                             join(model_bundle_dir, 'pipeline-config.json'))

        zip_path = join(self.output_dir_local, basename(self.model_bundle_uri))
        log.info(f'Saving bundle to {zip_path}.')
        zipdir(model_bundle_dir, zip_path)

    def _bundle_model(self, model_bundle_dir: str) -> None:
        """Save model weights and copy them to bundle dir.."""
        torch.save(self.model.state_dict(), self.last_model_weights_path)
        shutil.copyfile(self.last_model_weights_path,
                        join(model_bundle_dir, 'model.pth'))

    def _bundle_modules(self, model_bundle_dir: str) -> None:
        """Copy modules into bundle."""
        if isdir(self.modules_dir):
            log.info('Copying modules into bundle.')
            bundle_modules_dir = join(model_bundle_dir, MODULES_DIRNAME)
            if isdir(bundle_modules_dir):
                shutil.rmtree(bundle_modules_dir)
            shutil.copytree(self.modules_dir, bundle_modules_dir)

    def _bundle_transforms(self, model_bundle_dir: str) -> None:
        """Copy definition files for custom transforms, if any, into bundle.

        Copies definition files for custom albumentations transforms into
        bundle and changes the paths in the config to point to the new
        locations. The new paths are relative and will be automatically
        converted to full paths when loading from the bundle.
        """
        transforms = self.cfg.data.get_custom_albumentations_transforms()
        if len(transforms) == 0:
            return

        bundle_transforms_dir = join(model_bundle_dir, TRANSFORMS_DIRNAME)
        if isdir(bundle_transforms_dir):
            shutil.rmtree(bundle_transforms_dir)
        make_dir(bundle_transforms_dir)

        for tf in transforms:
            tf_bundle_path = download_or_copy(tf['lambda_transforms_path'],
                                              bundle_transforms_dir)
            # convert to a relative path
            tf['lambda_transforms_path'] = join('model-bundle',
                                                TRANSFORMS_DIRNAME,
                                                basename(tf_bundle_path))

    def get_start_epoch(self) -> int:
        """Get start epoch.

        If training was interrupted, this returns the last complete epoch + 1.
        """
        start_epoch = 0
        if isfile(self.log_path):
            with open(self.log_path) as log_file:
                lines = log_file.readlines()
                # if empty or containing only the header row
                if len(lines) <= 1:
                    return 0
                last_line = lines[-1]
            last_epoch = int(last_line.split(',')[0].strip())
            start_epoch = last_epoch + 1
        return start_epoch

    def load_init_weights(self,
                          model_weights_path: Optional[str] = None) -> None:
        """Load the weights to initialize model."""
        cfg = self.cfg
        uri = None
        args = {}

        if cfg.model is not None:
            uri = cfg.model.init_weights
            args['strict'] = cfg.model.load_strict

        if model_weights_path is not None:
            uri = model_weights_path

        if uri is None:
            return

        log.info(f'Loading model weights from: {uri}')
        self.load_weights(uri=uri, **args)

    def load_weights(self, uri: str, **kwargs) -> None:
        """Load model weights from a file."""
        weights_path = download_if_needed(uri)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device), **kwargs)

    def load_checkpoint(self):
        """Load last weights from previous run if available."""
        weights_path = self.last_model_weights_path
        if isfile(weights_path):
            log.info(f'Loading checkpoint from {weights_path}')
            args = {}
            if self.cfg.model is not None:
                args['strict'] = self.cfg.model.load_strict
            self.load_weights(uri=weights_path, **args)

    def to_device(self, x: Any, device: str) -> Any:
        """Load Tensors onto a device.

        Args:
            x: some object with Tensors in it
            device: 'cpu' or 'cuda'

        Returns:
            x but with any Tensors in it on the device
        """
        if isinstance(x, list):
            return [_x.to(device) if _x is not None else _x for _x in x]
        else:
            return x.to(device)

    def train_epoch(
            self,
            optimizer: 'Optimizer',
            step_scheduler: Optional['_LRScheduler'] = None) -> MetricDict:
        """Train for a single epoch."""
        start = time.time()
        self.model.train()
        num_samples = 0
        outputs = []
        with tqdm(self.train_dl, desc='Training') as bar:
            for batch_ind, (x, y) in enumerate(bar):
                x = self.to_device(x, self.device)
                y = self.to_device(y, self.device)
                batch = (x, y)
                optimizer.zero_grad()
                output = self.train_step(batch, batch_ind)
                output['train_loss'].backward()
                optimizer.step()
                # detach tensors in the output, if any, to avoid memory leaks
                for k, v in output.items():
                    output[k] = v.detach() if isinstance(v, Tensor) else v
                outputs.append(output)
                if step_scheduler is not None:
                    step_scheduler.step()
                num_samples += x.shape[0]
        if len(outputs) == 0:
            raise ValueError('Training dataset did not return any batches')
        metrics = self.train_end(outputs, num_samples)
        end = time.time()
        train_time = datetime.timedelta(seconds=end - start)
        metrics['train_time'] = str(train_time)
        return metrics

    def validate_epoch(self, dl: DataLoader) -> MetricDict:
        """Validate for a single epoch."""
        start = time.time()
        self.model.eval()
        num_samples = 0
        outputs = []
        with torch.inference_mode():
            with tqdm(dl, desc='Validating') as bar:
                for batch_ind, (x, y) in enumerate(bar):
                    x = self.to_device(x, self.device)
                    y = self.to_device(y, self.device)
                    batch = (x, y)
                    output = self.validate_step(batch, batch_ind)
                    outputs.append(output)
                    num_samples += x.shape[0]
        end = time.time()
        validate_time = datetime.timedelta(seconds=end - start)

        metrics = self.validate_end(outputs, num_samples)
        metrics['valid_time'] = str(validate_time)
        return metrics

    def overfit(self):
        """Optimize model using the same batch repeatedly."""
        self.on_overfit_start()

        x, y = next(iter(self.train_dl))
        x = self.to_device(x, self.device)
        y = self.to_device(y, self.device)
        batch = (x, y)

        num_steps = self.cfg.solver.overfit_num_steps
        with tqdm(range(num_steps), desc='Overfitting') as bar:
            for step in bar:
                loss = self.train_step(batch, step)['train_loss']
                loss.backward()
                self.opt.step()

                if (step + 1) % 25 == 0:
                    log.info('\nstep: {}'.format(step))
                    log.info('train_loss: {}'.format(loss))

        torch.save(self.model.state_dict(), self.last_model_weights_path)

    def train(self, epochs: Optional[int] = None):
        """Training loop that will attempt to resume training if appropriate."""
        start_epoch = self.get_start_epoch()

        if epochs is None:
            end_epoch = self.cfg.solver.num_epochs
        else:
            end_epoch = start_epoch + epochs

        if (start_epoch > 0 and start_epoch < end_epoch):
            log.info(f'Resuming training from epoch {start_epoch}')

        self.on_train_start()
        for epoch in range(start_epoch, end_epoch):
            log.info(f'epoch: {epoch}')
            train_metrics = self.train_epoch(
                optimizer=self.opt, step_scheduler=self.step_scheduler)
            if self.epoch_scheduler:
                self.epoch_scheduler.step()
            valid_metrics = self.validate_epoch(self.valid_dl)
            metrics = dict(epoch=epoch, **train_metrics, **valid_metrics)
            log.info(f'metrics:\n{pformat(metrics)}')

            self.on_epoch_end(epoch, metrics)

    def on_overfit_start(self):
        """Hook that is called at start of overfit routine."""
        pass

    def on_train_start(self):
        """Hook that is called at start of train routine."""
        pass

    def on_epoch_end(self, curr_epoch, metrics):
        """Hook that is called at end of epoch.

        Writes metrics to CSV and TB, and saves model.
        """
        if not isfile(self.log_path):
            with open(self.log_path, 'w') as log_file:
                log_writer = csv.writer(log_file)
                row = self.metric_names
                log_writer.writerow(row)

        with open(self.log_path, 'a') as log_file:
            log_writer = csv.writer(log_file)
            row = [metrics[k] for k in self.metric_names]
            log_writer.writerow(row)

        if self.cfg.log_tensorboard:
            for key, val in metrics.items():
                if isinstance(val, numbers.Number):
                    self.tb_writer.add_scalar(key, val, curr_epoch)
            self.tb_writer.flush()

        torch.save(self.model.state_dict(), self.last_model_weights_path)

        if (curr_epoch + 1) % self.cfg.solver.sync_interval == 0:
            self.sync_to_cloud()

    def eval_model(self, split: str):
        """Evaluate model using a particular dataset split.

        Gets validation metrics and saves them along with prediction plots.

        Args:
            split: the dataset split to use: train, valid, or test.
        """
        log.info(f'Evaluating on {split} set...')
        dl = self.get_dataloader(split)
        metrics = self.validate_epoch(dl)
        log.info(f'metrics: {metrics}')
        json_to_file(metrics,
                     join(self.output_dir_local, f'{split}_metrics.json'))
        self.plot_predictions(split, self.cfg.data.preview_batch_limit)
