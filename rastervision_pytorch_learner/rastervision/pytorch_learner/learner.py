from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List,
                    Literal, Optional, Tuple, Union, Type)
from abc import ABC, abstractmethod
from os.path import join, isfile, basename, isdir
import warnings
from time import perf_counter
import datetime
import shutil
import logging
from subprocess import Popen
import numbers
from pprint import pformat
import gc

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from rastervision.pipeline import rv_config_ as rv_config
from rastervision.pipeline.utils import get_env_var
from rastervision.pipeline.file_system import (
    sync_to_dir, json_to_file, make_dir, zipdir, download_if_needed,
    download_or_copy, sync_from_dir, get_local_path, unzip, is_local,
    get_tmp_dir)
from rastervision.pipeline.file_system.utils import file_exists
from rastervision.pipeline.utils import terminate_at_exit
from rastervision.pipeline.config import build_config
from rastervision.pytorch_learner.utils import (
    aggregate_metrics, DDPContextManager, get_hubconf_dir_from_cfg,
    get_learner_config_from_bundle_dir, log_metrics_to_csv, log_system_details,
    ONNXRuntimeAdapter)
from rastervision.pytorch_learner.dataset.visualizer import Visualizer

if TYPE_CHECKING:
    from typing import Self
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.utils.data import Dataset, Sampler
    from rastervision.pytorch_learner import LearnerConfig

warnings.filterwarnings('ignore')

CHECKPOINTS_DIRNAME = 'checkpoints'
MODULES_DIRNAME = 'modules'
TRANSFORMS_DIRNAME = 'custom_albumentations_transforms'
BUNDLE_MODEL_WEIGHTS_FILENAME = 'model.pth'
BUNDLE_MODEL_ONNX_FILENAME = 'model.onnx'

log = logging.getLogger(__name__)

MetricDict = Dict[str, float]


class Learner(ABC):
    """Abstract training and prediction routines for a model.

    This can be subclassed to handle different computer vision tasks.

    The datasets, model, optimizer, and schedulers will be generated from the
    :class:`.LearnerConfig` if not specified in the constructor.

    If instantiated with ``training=False``, the training apparatus (loss,
    optimizer, scheduler, logging, etc.) will not be set up and the model will
    be put into eval mode.

    .. note::

        This class supports distributed training via PyTorch DDP. If
        instantiated as a DDP process, it will automatically read WORLD_SIZE,
        RANK, and LOCAL_RANK environment variables. Alternatively, if
        ``RASTERVISION_USE_DDP=YES`` (the default), and multiple GPUs are
        detected, it will spawn DDP processes itself (one per GPU) when
        training. DDP options that may be set via environment variables or an
        INI file (see :ref:`raster vision config`) are:

        - ``RASTERVISION_USE_DDP``: Use DDP? Default: ``YES``.
        - ``RASTERVISION_DDP_BACKEND``: Default: ``nccl``.
        - ``RASTERVISION_DDP_START_METHOD``: One of ``spawn``, ``fork``, or
          ``forkserver``. Default: ``spawn``.

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
        self.training = training
        self._onnx_mode = (model_weights_path is not None
                           and model_weights_path.lower().endswith('.onnx'))
        if self.onnx_mode and self.training:
            raise ValueError('Training mode is not supported for ONNX models.')
        if model is None and cfg.model is None and not self.onnx_mode:
            raise ValueError(
                'cfg.model can only be None if a custom model is specified '
                'or if model_weights_path is an .onnx file.')

        if tmp_dir is None:
            self._tmp_dir = get_tmp_dir()
            tmp_dir = self._tmp_dir.name
        self.tmp_dir = tmp_dir

        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

        self.train_dl = None
        self.valid_dl = None
        self.test_dl = None

        self.model = model
        self.loss = loss
        self.opt = optimizer
        self.epoch_scheduler = epoch_scheduler
        self.step_scheduler = step_scheduler

        self.tb_process = None
        self.tb_writer = None
        self.tb_log_dir = None

        self.setup_ddp_params()

        if self.avoid_activating_cuda_runtime:
            device = 'cuda'
        else:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)

        # ---------------------------
        # Set URIs
        # ---------------------------
        self.output_dir = None
        self.output_dir_local = None
        self.model_bundle_uri = None
        self.modules_dir = None
        self.checkpoints_dir_local = None

        if self.training:
            if output_dir is None and cfg.output_uri is None:
                raise ValueError('output_dir or LearnerConfig.output_uri must '
                                 'be specified in training mode.')
            if output_dir is not None and cfg.output_uri is not None:
                log.warning(
                    'Both output_dir and LearnerConfig.output_uri specified. '
                    'LearnerConfig.output_uri will be ignored.')
            if output_dir is None:
                assert cfg.output_uri is not None
                self.output_dir = cfg.output_uri
                self.model_bundle_uri = cfg.get_model_bundle_uri()
            else:
                self.output_dir = output_dir
                self.model_bundle_uri = join(self.output_dir,
                                             'model-bundle.zip')
            if is_local(self.output_dir):
                self.output_dir_local = self.output_dir
                make_dir(self.output_dir_local)
            else:
                self.output_dir_local = get_local_path(self.output_dir,
                                                       tmp_dir)
                make_dir(self.output_dir_local, force_empty=True)
                if self.training:
                    self.sync_from_cloud()
                log.info(f'Local output dir: {self.output_dir_local}')
                log.info(f'Remote output dir: {self.output_dir}')

            self.modules_dir = join(self.output_dir, MODULES_DIRNAME)
            self.checkpoints_dir_local = join(self.output_dir_local,
                                              CHECKPOINTS_DIRNAME)
            make_dir(self.checkpoints_dir_local)

        # ---------------------------
        self.init_model_weights_path = model_weights_path
        self.init_model_def_path = model_def_path
        self.init_loss_def_path = loss_def_path

        if not self.distributed:
            self.setup_model(
                model_weights_path=model_weights_path,
                model_def_path=model_def_path)

        if self.training:
            self.setup_training(loss_def_path=loss_def_path)
            if self.model is not None:
                self.model.train()
        else:
            if not self.onnx_mode:
                self.model.eval()

        self.visualizer = self.get_visualizer_class()(
            cfg.data.class_names, cfg.data.class_colors,
            cfg.data.plot_options.transform,
            cfg.data.plot_options.channel_display_groups)

    @classmethod
    def from_model_bundle(cls: Type,
                          model_bundle_uri: str,
                          tmp_dir: Optional[str] = None,
                          cfg: Optional['LearnerConfig'] = None,
                          training: bool = False,
                          use_onnx_model: Optional[bool] = None,
                          **kwargs) -> 'Self':
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
            use_onnx_model (Optional[bool]): If True and training=False and a
                model.onnx file is available in the bundle, use that for
                inference rather than the PyTorch weights. Defaults to the
                boolean environment variable RASTERVISION_USE_ONNX if set,
                False otherwise.
            **kwargs: Extra args for :meth:`.__init__`.

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

        if cfg is None:
            cfg = get_learner_config_from_bundle_dir(model_bundle_dir)

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

        if use_onnx_model is None:
            use_onnx_model = rv_config.get_namespace_option(
                'rastervision', 'USE_ONNX', as_bool=True)
        onnx_mode = False
        if not training and use_onnx_model:
            onnx_path = join(model_bundle_dir, 'model.onnx')
            if file_exists(onnx_path):
                model_weights_path = onnx_path
                onnx_mode = True

        if not onnx_mode:
            if cfg.model is None and kwargs.get('model') is None:
                raise ValueError(
                    'Model definition is not saved in the model-bundle. '
                    'Please specify the model explicitly.')
            model_weights_path = join(model_bundle_dir,
                                      BUNDLE_MODEL_WEIGHTS_FILENAME)

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

    def main(self):
        """Main training sequence.

        This plots the dataset, runs a training and validation loop (which will
        resume if interrupted), logs stats, plots predictions, and syncs
        results to the cloud.
        """
        if self.distributed:
            with self.ddp():
                self._main()
        else:
            self._main()

    def _main(self):
        cfg = self.cfg
        if not self.is_ddp_process or self.is_ddp_local_master:
            if not self.avoid_activating_cuda_runtime:
                log_system_details()
            log.info(cfg)
        log.info(f'Using device: {self.device}')

        if not self.distributed:
            self.run_tensorboard()

        self.train()
        if cfg.save_model_bundle:
            if not self.is_ddp_process or self.is_ddp_master:
                self.save_model_bundle()
        self.stop_tensorboard()
        if cfg.eval_train:
            self.validate('train')
        self.validate('valid')
        if not self.is_ddp_process or self.is_ddp_master:
            self.sync_to_cloud()

    ###########################
    # Training and validation
    ###########################
    def train(self, epochs: Optional[int] = None):
        """Run training loop, resuming training if appropriate"""
        start_epoch, end_epoch = self.get_start_and_end_epochs(epochs)

        if start_epoch >= end_epoch:
            log.info('Training already completed. Skipping.')
            return

        if (start_epoch > 0 and start_epoch < end_epoch):
            log.info('Resuming training from epoch %d', start_epoch)

        if self.is_ddp_process:  # pragma: no cover
            self._run_train_distributed(self.ddp_rank, self.ddp_world_size,
                                        start_epoch, end_epoch)
        elif self.distributed:  # pragma: no cover
            log.info('Spawning %d DDP processes', self.ddp_world_size)
            mp.start_processes(
                self._run_train_distributed,
                args=(self.ddp_world_size, start_epoch, end_epoch),
                nprocs=self.ddp_world_size,
                join=True,
                start_method=self.ddp_start_method)
        else:
            self._train(start_epoch, end_epoch)

    def _train(self, start_epoch: int, end_epoch: int):  # pragma: no cover
        """Training loop."""
        self.on_train_start()
        for epoch in range(start_epoch, end_epoch):
            log.info(f'epoch: {epoch}')

            train_metrics = self.train_epoch(
                optimizer=self.opt, step_scheduler=self.step_scheduler)

            if self.epoch_scheduler:
                self.epoch_scheduler.step()

            valid_metrics = self.validate_epoch(self.valid_dl)

            metrics = dict(epoch=epoch, **train_metrics, **valid_metrics)
            log.info(f'metrics:\n{pformat(metrics, sort_dicts=False)}')

            self.on_epoch_end(epoch, metrics)

    def _train_distributed(self, start_epoch: int,
                           end_epoch: int):  # pragma: no cover
        """Distributed training loop."""
        if self.is_ddp_master:
            self.on_train_start()

        train_dl = self.build_dataloader('train', distributed=True)
        val_dl = self.build_dataloader('valid', distributed=True)
        for epoch in range(start_epoch, end_epoch):
            log.info(f'epoch: {epoch}')

            train_dl.sampler.set_epoch(epoch)

            train_metrics = self.train_epoch(
                optimizer=self.opt,
                step_scheduler=self.step_scheduler,
                dataloader=train_dl)

            valid_metrics = self.validate_epoch(val_dl)

            if self.is_ddp_master:
                metrics = dict(epoch=epoch, **train_metrics, **valid_metrics)
                log.info(f'metrics:\n{pformat(metrics, sort_dicts=False)}')
                self.on_epoch_end(epoch, metrics)

            if self.epoch_scheduler:
                self.epoch_scheduler.step()

            dist.barrier()

    def _run_train_distributed(self, rank: int, world_size: int,
                               *args):  # pragma: no cover
        """Method executed by each DDP worker."""
        with self.ddp(rank, world_size):
            self.setup_model(
                model_weights_path=self.init_model_weights_path,
                model_def_path=self.init_model_def_path)
            self.setup_training(self.init_loss_def_path)
            self._train_distributed(*args)

    def train_epoch(
            self,
            optimizer: 'Optimizer',
            dataloader: Optional[DataLoader] = None,
            step_scheduler: Optional['_LRScheduler'] = None) -> MetricDict:
        """Train for a single epoch."""
        self.model.train()
        if dataloader is None:
            dataloader = self.train_dl
        start = perf_counter()
        outputs = []
        if self.ddp_rank is not None:
            desc = f'Training (GPU={self.ddp_rank})'
        else:
            desc = 'Training'
        with tqdm(self.train_dl, desc=desc) as bar:
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
        if len(outputs) == 0:
            raise ValueError('Training dataset did not return any batches')
        metrics = self.train_end(outputs)
        end = perf_counter()
        train_time = datetime.timedelta(seconds=end - start)
        metrics['train_time'] = str(train_time)
        return metrics

    @abstractmethod
    def train_step(self, batch: Any, batch_ind: int) -> MetricDict:
        """Compute loss for a single training batch.

        Args:
            batch: batch data needed to compute loss
            batch_ind: index of batch within epoch

        Returns:
            dict with 'train_loss' as key and possibly other losses
        """

    def on_train_start(self):
        """Hook that is called at start of train routine."""
        self.log_data_stats()
        self.plot_dataloaders(self.cfg.data.preview_batch_limit)

    def train_end(self, outputs: List[Dict[str, Union[float, Tensor]]]
                  ) -> MetricDict:
        """Aggregate the output of train_step at the end of the epoch.

        Args:
            outputs: a list of outputs of train_step
        """
        metrics = aggregate_metrics(outputs)
        if self.is_ddp_process:
            metrics = self.reduce_distributed_metrics(metrics)
        return metrics

    def validate(self, split: Literal['train', 'valid', 'test'] = 'valid'):
        """Evaluate model on a particular data split."""
        if self.is_ddp_process:  # pragma: no cover
            self._run_validate_distributed(self.ddp_rank, self.ddp_world_size,
                                           split)
        elif self.distributed:  # pragma: no cover
            log.info('Spawning DDP processes')
            mp.start_processes(
                self._run_validate_distributed,
                args=(self.ddp_world_size, split),
                nprocs=self.ddp_world_size,
                join=True,
                start_method=self.ddp_start_method)
        else:
            self._validate(split)

    def _validate(self, split: Literal['train', 'valid', 'test'] = 'valid'
                  ):  # pragma: no cover
        """Evaluate model on a particular data split.

        Gets validation metrics and saves them along with prediction plots.

        Args:
            split: the dataset split to use: train, valid, or test.
        """
        log.info(f'Evaluating on {split} set...')
        dl = self.get_dataloader(split)
        if dl is None:
            self.setup_data()
        dl = self.get_dataloader(split)
        metrics = self.validate_epoch(dl)
        if self.is_ddp_process and not self.is_ddp_master:
            return
        log.info(f'metrics: {metrics}')
        json_to_file(metrics,
                     join(self.output_dir_local, f'{split}_metrics.json'))
        self.plot_predictions(split, self.cfg.data.preview_batch_limit)

    def _run_validate_distributed(self, rank: int, world_size: int,
                                  *args):  # pragma: no cover
        """Method executed by each DDP worker."""
        with self.ddp(rank, world_size):
            self.setup_model(
                model_weights_path=self.init_model_weights_path,
                model_def_path=self.init_model_def_path)
            self.setup_training(self.init_loss_def_path)
            self._validate(*args)

    def validate_epoch(self, dl: DataLoader) -> MetricDict:
        """Validate for a single epoch."""
        start = perf_counter()
        self.model.eval()
        outputs = []
        if self.ddp_rank is not None:
            desc = f'Validating (GPU={self.ddp_rank})'
        else:
            desc = 'Validating'
        with torch.inference_mode():
            with tqdm(dl, desc=desc) as bar:
                for batch_ind, (x, y) in enumerate(bar):
                    x = self.to_device(x, self.device)
                    y = self.to_device(y, self.device)
                    batch = (x, y)
                    output = self.validate_step(batch, batch_ind)
                    outputs.append(output)
        end = perf_counter()
        validate_time = datetime.timedelta(seconds=end - start)

        metrics = self.validate_end(outputs)
        metrics['valid_time'] = str(validate_time)
        return metrics

    @abstractmethod
    def validate_step(self, batch: Any, batch_ind: int) -> MetricDict:
        """Compute metrics on validation batch.

        Args:
            batch: batch data needed to compute validation metrics
            batch_ind: index of batch within epoch

        Returns:
            dict with metric names mapped to metric values
        """

    def validate_end(self, outputs: List[Dict[str, Union[float, Tensor]]]
                     ) -> MetricDict:
        """Aggregate the output of validate_step at the end of the epoch.

        Args:
            outputs: a list of outputs of validate_step
        """
        metrics = aggregate_metrics(outputs)
        if self.is_ddp_process:
            metrics = self.reduce_distributed_metrics(metrics)
        return metrics

    def on_epoch_end(self, curr_epoch: int, metrics: MetricDict) -> None:
        """Hook that is called at end of epoch.

        Writes metrics to CSV and TensorBoard, and saves model.
        """
        log_metrics_to_csv(self.log_path, metrics)

        if self.cfg.log_tensorboard:
            for key, val in metrics.items():
                if isinstance(val, numbers.Number):
                    self.tb_writer.add_scalar(key, val, curr_epoch)
            self.tb_writer.flush()

        if self.cfg.save_all_checkpoints and curr_epoch > 0:
            checkpoint_name = f'model-ckpt-epoch-{curr_epoch - 1}.pth'
            checkpoint_path = join(self.checkpoints_dir_local, checkpoint_name)
            shutil.move(self.last_model_weights_path, checkpoint_path)

        self.save_weights(self.last_model_weights_path)

        if (curr_epoch + 1) % self.cfg.solver.sync_interval == 0:
            self.sync_to_cloud()

    ########################
    # Prediction/inference
    ########################
    def predict(self, x: Tensor, raw_out: bool = False) -> Any:
        """Make prediction for an image or batch of images.

        Args:
            x (Tensor): Image or batch of images as a float Tensor with pixel
                values normalized to [0, 1].
            raw_out (bool): if True, return prediction probabilities

        Returns:
            The predictions, in probability form if raw_out is True, in
            class_id form otherwise.
        """
        x = self.to_batch(x).float()
        x = self.to_device(x, self.device)
        with torch.inference_mode():
            out = self.model(x)
            if not raw_out:
                out = self.prob_to_pred(self.post_forward(out))
        out = self.to_device(out, 'cpu')
        return out

    def predict_onnx(self, x: Tensor, raw_out: bool = False) -> Tensor:
        """Alternative to predict() for ONNX inference."""
        out = self.model(x)
        if not raw_out:
            out = self.prob_to_pred(self.post_forward(out))
        return out

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
            If return_format is 'z', the returned value is an iterator of
            whatever type the predictions are. Otherwise, the returned value is
            an iterator of tuples.
        """

        if return_format not in {'xyz', 'yz', 'z'}:
            raise ValueError('return_format must be one of "xyz", "yz", "z".')

        cfg = self.cfg

        num_workers = rv_config.get_namespace_option(
            'rastervision',
            'PREDICT_NUM_WORKERS',
            default=cfg.data.num_workers)

        dl_kw = dict(
            collate_fn=self.get_collate_fn(),
            batch_size=cfg.solver.batch_sz if cfg.solver else 1,
            num_workers=int(num_workers),
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

        if self.onnx_mode:
            log.info('Running inference with ONNX runtime.')
        else:
            self.model.eval()

        for x, y in dl:
            if self.onnx_mode:
                z = self.predict_onnx(x, raw_out=raw_out, **predict_kw)
            else:
                z = self.predict(x, raw_out=raw_out, **predict_kw)
                x = self.to_device(x, 'cpu')
            if batched_output:
                yield x, y, z
            else:
                for _x, _y, _z in zip(x, y, z):
                    yield _x, _y, _z

    def output_to_numpy(self, out: Tensor) -> np.ndarray:
        """Convert output of model to numpy format.

        Args:
            out: the output of the model in PyTorch format

        Returns: the output of the model in numpy format
        """
        return out.numpy()

    def prob_to_pred(self, x: Tensor) -> Tensor:
        """Convert a Tensor with prediction probabilities to class ids.

        The class ids should be the classes with the maximum probability.
        """
        raise NotImplementedError()

    #########
    # Setup
    #########
    def setup_ddp_params(self):
        """Set up and validate params related to PyTorch DDP."""

        ddp_allowed = rv_config.get_namespace_option(
            'rastervision', 'USE_DDP', True, as_bool=True)
        self.ddp_start_method = rv_config.get_namespace_option(
            'rastervision', 'DDP_START_METHOD', 'spawn').lower()

        self.is_ddp_process = False
        self.is_ddp_master = False
        self.is_ddp_local_master = False
        self.avoid_activating_cuda_runtime = False

        self.ddp_world_size = get_env_var('WORLD_SIZE', None, int)
        self.ddp_rank = get_env_var('RANK', None, int)
        self.ddp_local_rank = get_env_var('LOCAL_RANK', None, int)
        ddp_vars_set = all(
            v is not None
            for v in [self.ddp_world_size, self.ddp_rank, self.ddp_local_rank])

        if not ddp_allowed or not self.training:
            self.distributed = False
        elif ddp_vars_set:  # pragma: no cover
            self.distributed = True
            self.is_ddp_process = True
            self.is_ddp_master = self.ddp_rank == 0
            self.is_ddp_local_master = self.ddp_local_rank == 0
        elif self.ddp_start_method != 'spawn':
            # If ddp_start_method is "fork" or "forkserver", the CUDA runtime
            # must not be initialized before the fork; otherwise, a
            # "RuntimeError: Cannot re-initialize CUDA in forked subprocess."
            # error will be raised. We can avoid initializing it by not
            # calling any torch.cuda functions or creating tensors on the GPU.
            if self.ddp_world_size is None:
                raise ValueError(
                    'WORLD_SIZE env variable must be specified if '
                    'RASTERVISION_DDP_START_METHOD is not "spawn".')
            self.distributed = True
            self.avoid_activating_cuda_runtime = True
        elif torch.cuda.is_available():
            dist_available = dist.is_available()
            gpu_count = torch.cuda.device_count()
            multi_gpus = gpu_count > 1
            self.distributed = ddp_allowed and dist_available and multi_gpus
            if self.distributed:
                log.info(
                    'Multiple GPUs detected (%d), will use DDP for training.',
                    gpu_count)
                world_size_is_set = self.ddp_world_size is not None
                if not world_size_is_set:
                    self.ddp_world_size = gpu_count
                if world_size_is_set and self.ddp_world_size < gpu_count:
                    log.info('Using only WORLD_SIZE=%d of total %d GPUs.',
                             self.ddp_world_size, gpu_count)
        else:
            self.distributed = False

        if not self.distributed:
            return

        # pragma: no cover
        if self.model is not None:
            raise ValueError(
                'In distributed mode, the model must be specified via '
                'ModelConfig in LearnerConfig rather than be passed '
                'as an instantiated object.')

        dses_passed = any([self.train_ds, self.valid_ds, self.test_ds])
        if dses_passed and self.ddp_start_method != 'fork':
            raise ValueError(
                'In distributed mode, if '
                'RASTERVISION_DDP_START_METHOD != "fork", datasets must be '
                'specified via DataConfig in LearnerConfig rather than be '
                'passed as instantiated objects.')

        if self.ddp_local_rank is not None:
            self.device = torch.device('cuda', self.ddp_local_rank)

        log.info('Using DDP')
        log.info(f'World size: {self.ddp_world_size}')
        log.info(f'DDP start method: {self.ddp_start_method}')
        if self.is_ddp_process:
            log.info(f'DDP rank: {self.ddp_rank}')
            log.info(f'DDP local rank: {self.ddp_local_rank}')

    def setup_training(self, loss_def_path: Optional[str] = None) -> None:
        """Set up model, data, loss, optimizers and various paths.

        The exact behavior differs based on whether this method is called in
        a distributed scenario.

        Args:
            loss_def_path: A local path to a directory with a ``hubconf.py``. If
                provided, the loss function definition is imported from here.
                This is used when loading an external loss function from a
                model-bundle. Defaults to ``None``.
        """
        cfg = self.cfg

        self.config_path = join(self.output_dir_local, 'learner-config.json')
        cfg.to_file(self.config_path)
        self.log_path = join(self.output_dir_local, 'log.csv')
        self.last_model_weights_path = join(self.output_dir_local,
                                            'last-model.pth')

        if not self.distributed:
            # data
            self.setup_data()
            # model
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
            return

        # DDP
        if self.is_ddp_process and dist.is_initialized():  # pragma: no cover
            # model
            if self.model is not None:
                self.load_checkpoint()
            # data
            self.setup_data()
            # optimization
            start_epoch = self.get_start_epoch()
            self.setup_loss(loss_def_path=loss_def_path)
            if self.opt is None:
                self.opt = self.build_optimizer()
            if self.step_scheduler is None:
                self.step_scheduler = self.build_step_scheduler(start_epoch)
            if self.epoch_scheduler is None:
                self.epoch_scheduler = self.build_epoch_scheduler(start_epoch)

            if self.is_ddp_master:
                self.setup_tensorboard()
        else:  # pragma: no cover
            if self.ddp_start_method == 'fork':
                self.setup_data()

    def get_start_and_end_epochs(
            self, epochs: Optional[int] = None) -> Tuple[int, int]:
        """Get start and end epochs given epochs."""
        start_epoch = self.get_start_epoch()
        if epochs is None:
            end_epoch = self.cfg.solver.num_epochs
        else:
            end_epoch = start_epoch + epochs
        return start_epoch, end_epoch

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
        if self.onnx_mode:
            self.model = self.load_onnx_model(model_weights_path)
            return
        if self.model is None:
            self.model = self.build_model(model_def_path=model_def_path)
        self.model.to(device=self.device)
        if self.is_ddp_process:  # pragma: no cover
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        self.load_init_weights(model_weights_path=model_weights_path)

    def build_model(self, model_def_path: Optional[str] = None) -> nn.Module:
        """Build a PyTorch model."""
        cfg = self.cfg

        in_channels = cfg.data.img_channels
        if in_channels is None:
            log.warning('DataConfig.img_channels is None. Defaulting to 3.')
            in_channels = 3

        model = cfg.model.build(
            num_classes=cfg.data.num_classes,
            in_channels=in_channels,
            save_dir=self.modules_dir,
            hubconf_dir=model_def_path,
            ddp_rank=self.ddp_local_rank)
        return model

    def setup_data(self, distributed: Optional[bool] = None):
        """Set datasets and dataLoaders for train, validation, and test sets.
        """
        if distributed is None:
            distributed = self.distributed

        if self.train_ds is None or self.valid_ds is None:
            if distributed:  # pragma: no cover
                if self.is_ddp_local_master:
                    train_ds, valid_ds, test_ds = self.build_datasets()
                    log.debug(f'{self.ddp_rank=} Done.')
                else:
                    log.debug(f'{self.ddp_rank=} Waiting.')
                dist.barrier()
                if not self.is_ddp_local_master:
                    train_ds, valid_ds, test_ds = self.build_datasets()
                    log.debug(f'{self.ddp_rank=} Done.')
                else:
                    log.debug(f'{self.ddp_rank=} Waiting.')
                dist.barrier()
            else:
                train_ds, valid_ds, test_ds = self.build_datasets()

            if self.train_ds is None:
                self.train_ds = train_ds
            if self.valid_ds is None:
                self.valid_ds = valid_ds
            if self.test_ds is None:
                self.test_ds = test_ds

        log.info('Building dataloaders')
        self.train_dl, self.valid_dl, self.test_dl = self.build_dataloaders(
            distributed=distributed)

    def build_datasets(self) -> Tuple['Dataset', 'Dataset', 'Dataset']:
        """Build Datasets for train, validation, and test splits."""
        log.info(f'Building datasets ...')
        train_ds, val_ds, test_ds = self.cfg.data.build(tmp_dir=self.tmp_dir)
        return train_ds, val_ds, test_ds

    def build_dataset(self,
                      split: Literal['train', 'valid', 'test']) -> 'Dataset':
        """Build Dataset for split."""
        log.info('Building %s dataset ...', split)
        ds = self.cfg.data.build_dataset(split=split, tmp_dir=self.tmp_dir)
        return ds

    def build_dataloaders(self, distributed: Optional[bool] = None
                          ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Build DataLoaders for train, validation, and test splits."""
        if distributed is None:
            distributed = self.distributed

        train_dl = self.build_dataloader('train', distributed=distributed)
        val_dl = self.build_dataloader('valid', distributed=distributed)

        test_dl = None
        if self.test_ds is not None and len(self.test_ds) > 0:
            test_dl = self.build_dataloader('test', distributed=distributed)

        return train_dl, val_dl, test_dl

    def build_dataloader(self,
                         split: Literal['train', 'valid', 'test'],
                         distributed: Optional[bool] = None,
                         **kwargs) -> DataLoader:
        """Build DataLoader for split."""
        if distributed is None:
            distributed = self.distributed

        ds = self.get_dataset(split)
        if ds is None:
            ds = self.build_dataset(split)

        batch_sz = self.cfg.solver.batch_sz
        num_workers = self.cfg.data.num_workers
        collate_fn = self.get_collate_fn()
        sampler = self.build_sampler(ds, split, distributed=distributed)

        if distributed:  # pragma: no cover
            world_sz = self.ddp_world_size
            if world_sz is None:
                raise ValueError('World size not set. '
                                 'Cannot determine per-process batch size.')
            if world_sz > batch_sz:
                raise ValueError(f'World size ({world_sz}) is greater '
                                 f'than total batch size ({batch_sz}).')
            batch_sz //= world_sz
            log.debug('Per GPU batch size: %d', batch_sz)

        args = dict(
            batch_size=batch_sz,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            multiprocessing_context='fork' if distributed else None,
        )
        args.update(**kwargs)

        if sampler is not None:
            args['sampler'] = sampler
        else:
            if split == 'train':
                args['shuffle'] = True
                # batchnorm layers expect batch size > 1 during training
                args['drop_last'] = (len(ds) % batch_sz) == 1
            else:
                args['shuffle'] = False

        dl = DataLoader(ds, **args)
        return dl

    def get_collate_fn(self) -> Optional[callable]:
        """Returns a custom collate_fn to use in DataLoader.

        None is returned if default collate_fn should be used.

        See https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        """
        return None

    def build_sampler(self,
                      ds: 'Dataset',
                      split: Literal['train', 'valid', 'test'],
                      distributed: bool = False) -> Optional['Sampler']:
        """Build an optional sampler for the split's dataloader."""
        split = split.lower()
        sampler = None
        if split == 'train':
            if distributed:  # pragma: no cover
                sampler = DistributedSampler(
                    ds,
                    shuffle=True,
                    num_replicas=self.ddp_world_size,
                    rank=self.ddp_rank)
        elif split == 'valid':
            if distributed:  # pragma: no cover
                sampler = DistributedSampler(
                    ds,
                    shuffle=False,
                    num_replicas=self.ddp_world_size,
                    rank=self.ddp_rank)
        return sampler

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

    ################
    # Visualization
    ################
    @abstractmethod
    def get_visualizer_class(self) -> Type[Visualizer]:
        """Returns a Visualizer class object for plotting data samples."""

    def plot_predictions(self,
                         split: Literal['train', 'valid', 'test'],
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

    #########
    # Bundle
    #########
    def save_model_bundle(self, export_onnx: bool = True):
        """Save a model bundle.

        This is a zip file with the model weights in .pth format and a serialized
        copy of the LearningConfig, which allows for making predictions in the future.
        """
        if self.cfg.model is None:
            log.warning(
                'Model was not configured via ModelConfig, and therefore, '
                'will not be reconstructable form the model-bundle. You will '
                'need to initialize the model yourself and pass it to '
                'from_model_bundle().')

        log.info('Creating bundle.')
        model_bundle_dir = join(self.tmp_dir, 'model-bundle')
        make_dir(model_bundle_dir, force_empty=True)

        self._bundle_model(model_bundle_dir, export_onnx=export_onnx)
        self._bundle_modules(model_bundle_dir)
        self._bundle_transforms(model_bundle_dir)

        cfg_uri = join(model_bundle_dir, 'learner-config.json')
        shutil.copy(self.config_path, cfg_uri)

        zip_path = join(self.output_dir_local, basename(self.model_bundle_uri))
        log.info(f'Saving bundle to {zip_path}.')
        zipdir(model_bundle_dir, zip_path)

    def _bundle_model(self, model_bundle_dir: str,
                      export_onnx: bool = True) -> None:
        """Save model weights and copy them to bundle dir."""
        model_not_set = self.model is None
        if model_not_set:
            self.model = self.build_model(self.init_model_def_path).cpu()
            self.load_checkpoint()

        path = join(model_bundle_dir, BUNDLE_MODEL_WEIGHTS_FILENAME)
        if file_exists(self.last_model_weights_path):
            shutil.copyfile(self.last_model_weights_path, path)
        else:
            self.save_weights(path)

        # ONNX
        if export_onnx:
            path = join(model_bundle_dir, BUNDLE_MODEL_ONNX_FILENAME)
            self.export_to_onnx(path)

        if model_not_set:
            self.model = None
            gc.collect()

    def export_to_onnx(self,
                       path: str,
                       model: Optional['nn.Module'] = None,
                       sample_input: Optional[Tensor] = None,
                       validate_export: bool = True,
                       **kwargs) -> None:
        """Export model to ONNX format via :func:`torch.onnx.export`.

        Args:
            path (str): File path to save to.
            model (Optional[nn.Module]): The model to export. If None,
                self.model will be used. Defaults to None.
            sample_input (Optional[Tensor]): Sample input to the model. If
                None, a single batch from any available DataLoader in this
                Learner will be used. Defaults to None.
            validate_export (bool): If True, use
                :func:`onnx.checker.check_model` to validate exported model.
                An exception is raised if the check fails. Defaults to True.
            **kwargs (dict): Keyword args to pass to :func:`torch.onnx.export`.
                These override the default values used in the function
                definition.

        Raises:
            ValueError: If sample_input is None and the Learner has no valid
                DataLoaders.
        """
        if model is None:
            model = self.model

        if isinstance(model, DDP):
            model = model.module

        training_state = model.training

        model.eval()
        if sample_input is None:
            dl = self.valid_dl
            if dl is None:
                dl = self.build_dataloader(
                    'valid', batch_size=1, num_workers=0, distributed=False)
            sample_input, _ = next(iter(dl))

        model_device = next(model.parameters()).device
        if model_device.type == 'cuda':
            torch.cuda.empty_cache()
        sample_input = self.to_device(sample_input, model_device)

        args = dict(
            input_names=['x'],
            output_names=['out'],
            dynamic_axes={
                'x': {
                    0: 'batch_size',
                    2: 'height',
                    3: 'width',
                },
                'out': {
                    0: 'batch_size',
                },
            },
            training=torch.onnx.TrainingMode.EVAL,
            opset_version=15,
        )
        args.update(**kwargs)
        log.info('Exporting to model to ONNX.')
        torch.onnx.export(model, sample_input, path, **args)

        model.train(training_state)

        if validate_export:
            import onnx
            model_onnx = onnx.load(path)
            onnx.checker.check_model(model_onnx)

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

    #########
    # Misc.
    #########
    def ddp(self, rank: Optional[int] = None, world_size: Optional[int] = None
            ) -> DDPContextManager:  # pragma: no cover
        """Return a :class:`DDPContextManager`.

        This should be used to wrap code that needs to be executed in parallel.
        It is safe call this recursively; recusive calls will have no affect.

        Note that :class:`DDPContextManager` does not start processes itself,
        but merely initializes and destroyes DDP process groups.

        Usage:

        .. code-block:: python

            with learner.ddp([rank], [world_size]):
                ...

        """
        if not self.distributed:
            raise ValueError('self.distributed is False')
        return DDPContextManager(self, rank, world_size)

    def reduce_distributed_metrics(self, metrics: dict):  # pragma: no cover
        """Average numeric metrics across processes."""
        for k in metrics.keys():
            v = metrics[k]
            if isinstance(v, (float, int)):
                v = torch.tensor(v, device=self.device)
            if isinstance(v, Tensor):
                dist.reduce(v, dst=0, op=dist.ReduceOp.SUM)
                if self.is_ddp_master:
                    metrics[k] = (v / self.ddp_world_size).item()
        return metrics

    def post_forward(self, x: Any) -> Any:
        """Post process output of call to model().

        Useful for when predictions are inside a structure returned by model().
        """
        return x

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

    def to_device(self, x: Any, device: Union[str, torch.device]) -> Any:
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

    def get_dataset(self, split: Literal['train', 'valid', 'test']
                    ) -> Optional[DataLoader]:
        """Get the Dataset for a split.

        Args:
            split: a split name which can be train, valid, or test
        """
        if split == 'train':
            return self.train_ds
        if split == 'valid':
            return self.valid_ds
        if split == 'test':
            return self.test_ds
        raise ValueError(f'{split} is not a valid split')

    def get_dataloader(self,
                       split: Literal['train', 'valid', 'test']) -> DataLoader:
        """Get the DataLoader for a split.

        Args:
            split: a split name which can be train, valid, or test
        """
        if split == 'train':
            return self.train_dl
        if split == 'valid':
            return self.valid_dl
        if split == 'test':
            return self.test_dl
        raise ValueError(f'{split} is not a valid split')

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

    def save_weights(self, path: str):
        """Save model weights to a local file."""
        model = self.model
        if isinstance(model, DDP):
            model = model.module
        torch.save(model.state_dict(), path)

    def load_weights(self, uri: str, **kwargs) -> None:
        """Load model weights from a file.

        Args:
            uri (str): URI.
            **kwargs: Extra args for :meth:`nn.Module.load_state_dict`.
        """
        weights_path = download_if_needed(uri)
        model = self.model
        if isinstance(model, DDP):
            model = model.module
        model.load_state_dict(
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

    def load_onnx_model(self, model_path: str) -> ONNXRuntimeAdapter:
        log.info(f'Loading ONNX model from {model_path}')
        path = download_if_needed(model_path)
        onnx_model = ONNXRuntimeAdapter.from_file(path)
        return onnx_model

    def log_data_stats(self):
        """Log stats about each DataSet."""
        if self.train_ds is not None:
            log.info(f'train_ds: {len(self.train_ds)} items')
        if self.valid_ds is not None:
            log.info(f'valid_ds: {len(self.valid_ds)} items')
        if self.test_ds is not None:
            log.info(f'test_ds: {len(self.test_ds)} items')

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
        if self.cfg.run_tensorboard:  # pragma: no cover
            log.info('Starting tensorboard process')
            self.tb_process = Popen(
                ['tensorboard', '--bind_all', f'--logdir={self.tb_log_dir}'])
            terminate_at_exit(self.tb_process)

    def stop_tensorboard(self):
        """Stop TB logging and server if it's running."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.tb_process is not None:  # pragma: no cover
            self.tb_process.terminate()

    @property
    def onnx_mode(self) -> bool:
        return self._onnx_mode
