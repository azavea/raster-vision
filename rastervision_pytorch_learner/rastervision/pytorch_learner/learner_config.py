from typing import (TYPE_CHECKING, Any, Iterable, Literal, Sequence)
from collections.abc import Callable
import os
from os.path import join, isdir
from enum import Enum
import random
import uuid
import logging

from typing_extensions import Annotated
from pydantic import (NonNegativeInt as NonNegInt, PositiveFloat, PositiveInt
                      as PosInt, StringConstraints)
from pydantic.v1.utils import sequence_like
import albumentations as A
import torch
from torch import (nn, optim)
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR, _LRScheduler
from torch.utils.data import Dataset, ConcatDataset, Subset

from rastervision.pipeline.config import (Config, register_config, ConfigError,
                                          Field, field_validator,
                                          model_validator)
from rastervision.pipeline.file_system import (list_paths, download_if_needed,
                                               unzip, file_exists,
                                               get_local_path, sync_from_dir)
from rastervision.core.data import (ClassConfig, Scene, DatasetConfig as
                                    SceneDatasetConfig)
from rastervision.core.rv_pipeline import (WindowSamplingConfig)
from rastervision.core.utils import NonEmptyStr, Proportion
from rastervision.pytorch_learner.utils import (
    validate_albumentation_transform, MinMaxNormalize,
    deserialize_albumentation_transform, get_hubconf_dir_from_cfg,
    torch_hub_load_local, torch_hub_load_github, torch_hub_load_uri)

if TYPE_CHECKING:
    from typing import Self
    from rastervision.core.data import SceneConfig
    from rastervision.pytorch_learner.learner import Learner

log = logging.getLogger(__name__)

default_augmentors = ['RandomRotate90', 'HorizontalFlip', 'VerticalFlip']
augmentors = [
    'Blur', 'RandomRotate90', 'HorizontalFlip', 'VerticalFlip', 'GaussianBlur',
    'GaussNoise', 'RGBShift', 'ToGray'
]

# types
RGBTuple = tuple[int, int, int]
ChannelInds = Sequence[NonNegInt]


class Backbone(str, Enum):
    alexnet = 'alexnet'
    densenet121 = 'densenet121'
    densenet169 = 'densenet169'
    densenet201 = 'densenet201'
    densenet161 = 'densenet161'
    googlenet = 'googlenet'
    inception_v3 = 'inception_v3'
    mnasnet0_5 = 'mnasnet0_5'
    mnasnet0_75 = 'mnasnet0_75'
    mnasnet1_0 = 'mnasnet1_0'
    mnasnet1_3 = 'mnasnet1_3'
    mobilenet_v2 = 'mobilenet_v2'
    resnet18 = 'resnet18'
    resnet34 = 'resnet34'
    resnet50 = 'resnet50'
    resnet101 = 'resnet101'
    resnet152 = 'resnet152'
    resnext50_32x4d = 'resnext50_32x4d'
    resnext101_32x8d = 'resnext101_32x8d'
    wide_resnet50_2 = 'wide_resnet50_2'
    wide_resnet101_2 = 'wide_resnet101_2'
    shufflenet_v2_x0_5 = 'shufflenet_v2_x0_5'
    shufflenet_v2_x1_0 = 'shufflenet_v2_x1_0'
    shufflenet_v2_x1_5 = 'shufflenet_v2_x1_5'
    shufflenet_v2_x2_0 = 'shufflenet_v2_x2_0'
    squeezenet1_0 = 'squeezenet1_0'
    squeezenet1_1 = 'squeezenet1_1'
    vgg11 = 'vgg11'
    vgg11_bn = 'vgg11_bn'
    vgg13 = 'vgg13'
    vgg13_bn = 'vgg13_bn'
    vgg16 = 'vgg16'
    vgg16_bn = 'vgg16_bn'
    vgg19_bn = 'vgg19_bn'
    vgg19 = 'vgg19'

    @staticmethod
    def int_to_str(x):
        mapping = {
            1: 'alexnet',
            2: 'densenet121',
            3: 'densenet169',
            4: 'densenet201',
            5: 'densenet161',
            6: 'googlenet',
            7: 'inception_v3',
            8: 'mnasnet0_5',
            9: 'mnasnet0_75',
            10: 'mnasnet1_0',
            11: 'mnasnet1_3',
            12: 'mobilenet_v2',
            13: 'resnet18',
            14: 'resnet34',
            15: 'resnet50',
            16: 'resnet101',
            17: 'resnet152',
            18: 'resnext50_32x4d',
            19: 'resnext101_32x8d',
            20: 'wide_resnet50_2',
            21: 'wide_resnet101_2',
            22: 'shufflenet_v2_x0_5',
            23: 'shufflenet_v2_x1_0',
            24: 'shufflenet_v2_x1_5',
            25: 'shufflenet_v2_x2_0',
            26: 'squeezenet1_0',
            27: 'squeezenet1_1',
            28: 'vgg11',
            29: 'vgg11_bn',
            30: 'vgg13',
            31: 'vgg13_bn',
            32: 'vgg16',
            33: 'vgg16_bn',
            34: 'vgg19_bn',
            35: 'vgg19'
        }
        return mapping[x]


@register_config('external-module')
class ExternalModuleConfig(Config):
    """Config describing an object to be loaded via Torch Hub."""
    uri: NonEmptyStr | None = Field(
        None,
        description=('Local uri of a zip file, or local uri of a directory,'
                     'or remote uri of zip file.'))
    github_repo: (
        Annotated[str,
                  StringConstraints(strip_whitespace=True, pattern=r'.+/.+')]
        | None) = Field(
            None, description='<repo-owner>/<repo-name>[:tag]')
    name: NonEmptyStr | None = Field(
        None,
        description=
        'Name of the folder in which to extract/copy the definition files.')
    entrypoint: NonEmptyStr = Field(
        ...,
        description=('Name of a Callable present in ``hubconf.py``. '
                     'See docs for ``torch.hub`` for details.'))
    entrypoint_args: list = Field(
        [],
        description='Args to pass to the entrypoint. Must be serializable.')
    entrypoint_kwargs: dict = Field(
        {},
        description=
        'Keyword args to pass to the entrypoint. Must be serializable.')
    force_reload: bool = Field(
        False, description='Force reload of module definition.')

    @model_validator(mode='after')
    def check_either_uri_or_repo(self) -> 'Self':
        has_uri = self.uri is not None
        has_repo = self.github_repo is not None
        if has_uri == has_repo:
            raise ConfigError(
                'Must specify one (and only one) of github_repo and uri.')
        return self

    def build(self,
              save_dir: str,
              hubconf_dir: str | None = None,
              ddp_rank: int | None = None) -> Any:
        """Load an external module via torch.hub.

        Note: Loading a PyTorch module is the typical use case, but there are
        no type restrictions on the object loaded through torch.hub.

        Args:
            save_dir (str): The module def will be saved here.
            hubconf_dir (str): Path to existing definition.
                If provided, the definition will not be fetched from the
                external source but instead from this dir. Defaults to None.

        Returns:
            The module loaded via torch.hub.
        """
        if hubconf_dir is not None:
            log.info(f'Using existing module definition at: {hubconf_dir}')
            module = torch_hub_load_local(
                hubconf_dir=hubconf_dir,
                entrypoint=self.entrypoint,
                *self.entrypoint_args,
                **self.entrypoint_kwargs)
            return module

        dst_dir = get_hubconf_dir_from_cfg(self, parent=save_dir)
        if ddp_rank is not None:
            # avoid conflicts when downloading
            os.environ['TORCH_HOME'] = f'~/.cache/torch/{ddp_rank}'
            if ddp_rank != 0:
                dst_dir = None

        if self.github_repo is not None:
            log.info(f'Fetching module definition from: {self.github_repo}')
            module = torch_hub_load_github(
                repo=self.github_repo,
                entrypoint=self.entrypoint,
                *self.entrypoint_args,
                dst_dir=dst_dir,
                **self.entrypoint_kwargs)
        else:
            log.info(f'Fetching module definition from: {self.uri}')
            module = torch_hub_load_uri(
                uri=self.uri,
                entrypoint=self.entrypoint,
                *self.entrypoint_args,
                dst_dir=dst_dir,
                **self.entrypoint_kwargs)
        return module


def model_config_upgrader(cfg_dict, version):
    if version == 0:
        cfg_dict['backbone'] = Backbone.int_to_str(cfg_dict['backbone'])
    return cfg_dict


@register_config('model', upgrader=model_config_upgrader)
class ModelConfig(Config):
    """Config related to models."""
    backbone: Backbone = Field(
        Backbone.resnet18,
        description='The torchvision.models backbone to use.')
    pretrained: bool = Field(
        True,
        description=(
            'If True, use ImageNet weights. If False, use random initialization.'
        ))
    init_weights: str | None = Field(
        None,
        description=('URI of PyTorch model weights used to initialize model. '
                     'If set, this supersedes the pretrained option.'))
    load_strict: bool = Field(
        True,
        description=(
            'If True, the keys in the state dict referenced by init_weights '
            'must match exactly. Setting this to False can be useful if you '
            'just want to load the backbone of a model.'))
    external_def: ExternalModuleConfig | None = Field(
        None,
        description='If specified, the model will be built from the '
        'definition from this external source, using Torch Hub.')
    extra_args: dict = Field(
        {},
        description='Other implementation-specific args that might be useful '
        'for constructing the default model. This is ignored if using an '
        'external model.')

    def get_backbone_str(self):
        return self.backbone.name

    def build(self,
              num_classes: int,
              in_channels: int,
              save_dir: str | None = None,
              hubconf_dir: str | None = None,
              ddp_rank: int | None = None,
              **kwargs) -> nn.Module:
        """Build and return a model based on the config.

        Args:
            num_classes (int): Number of classes.
            in_channels (int): Number of channels in the images that
                will be fed into the model. Defaults to 3.
            save_dir (str|None): Used for building external_def
                if specified. Defaults to None.
            hubconf_dir (str|None): Used for building
                external_def if specified. Defaults to None.
            **kwargs: Extra args for :meth:`.build_default_model`.

        Returns:
            A PyTorch nn.Module.
        """
        if self.external_def is not None:
            return self.build_external_model(
                save_dir=save_dir, hubconf_dir=hubconf_dir, ddp_rank=ddp_rank)
        return self.build_default_model(num_classes, in_channels, **kwargs)

    def build_default_model(self, num_classes: int, in_channels: int,
                            **kwargs) -> nn.Module:
        """Build and return the default model.

        Args:
            num_classes (int): Number of classes.
            in_channels (int): Number of channels in the images that
                will be fed into the model. Defaults to 3.

        Returns:
            A PyTorch nn.Module.
        """
        raise NotImplementedError()

    def build_external_model(self,
                             save_dir: str,
                             hubconf_dir: str | None = None,
                             ddp_rank: int | None = None) -> nn.Module:
        """Build and return an external model.

        Args:
            save_dir (str): The module def will be saved here.
            hubconf_dir (str|None): Path to existing definition.
                Defaults to None.

        Returns:
            A PyTorch nn.Module.
        """
        return self.external_def.build(
            save_dir, hubconf_dir=hubconf_dir, ddp_rank=ddp_rank)


def solver_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 3:
        # 'ignore_last_class' replaced by 'ignore_class_index' in version 4
        ignore_last_class = cfg_dict.get('ignore_last_class')
        if ignore_last_class is not None:
            if ignore_last_class is not False:
                cfg_dict['ignore_class_index'] = -1
            del cfg_dict['ignore_last_class']
    if version == 4:
        # removed in version 5
        cfg_dict.pop('test_batch_sz', None)
        cfg_dict.pop('test_num_epochs', None)
        cfg_dict.pop('overfit_num_steps', None)
    return cfg_dict


@register_config('solver', upgrader=solver_config_upgrader)
class SolverConfig(Config):
    """Config related to solver aka optimizer."""
    lr: PositiveFloat = Field(1e-4, description='Learning rate.')
    num_epochs: PosInt = Field(
        10,
        description=
        'Number of epochs (ie. sweeps through the whole training set).')
    sync_interval: PosInt = Field(
        1, description='The interval in epochs for each sync to the cloud.')
    batch_sz: PosInt = Field(32, description='Batch size.')
    one_cycle: bool = Field(
        True,
        description=
        ('If True, use triangular LR scheduler with a single cycle across all '
         'epochs with start and end LR being lr/10 and the peak being lr.'))
    multi_stage: list[int] = Field(
        [], description=('List of epoch indices at which to divide LR by 10.'))
    class_loss_weights: Sequence[float] | None = Field(
        None, description=('Class weights for weighted loss.'))
    ignore_class_index: int | None = Field(
        None,
        description='If specified, this index is ignored when computing the '
        'loss. See pytorch documentation for nn.CrossEntropyLoss for more '
        'details. This can also be negative, in which case it is treated as a '
        'negative slice index i.e. -1 = last index, -2 = second-last index, '
        'and so on.')
    external_loss_def: ExternalModuleConfig | None = Field(
        None,
        description='If specified, the loss will be built from the definition '
        'from this external source, using Torch Hub.')

    @model_validator(mode='after')
    def check_no_loss_opts_if_external(self) -> 'Self':
        has_external_loss_def = self.external_loss_def is not None
        has_ignore_class_index = self.ignore_class_index is not None
        has_class_loss_weights = self.class_loss_weights is not None

        if has_external_loss_def:
            if has_ignore_class_index:
                raise ConfigError('ignore_class_index is not supported '
                                  'with external_loss_def.')
            if has_class_loss_weights:
                raise ConfigError('class_loss_weights is not supported '
                                  'with external_loss_def.')
        return self

    def build_loss(
            self,
            num_classes: int,
            save_dir: str | None = None,
            hubconf_dir: str | None = None) -> Callable[..., torch.Tensor]:
        """Build and return a loss function based on the config.

        Args:
            num_classes: Number of classes.
            save_dir: Used for building ``external_loss_def`` if specified.
                Defaults to ``None``.
            hubconf_dir: Used for building ``external_loss_def`` if specified.
                Defaults to ``None``.

        Returns:
            Loss function.
        """
        if self.external_loss_def is not None:
            return self.external_loss_def.build(
                save_dir=save_dir, hubconf_dir=hubconf_dir)

        args = {}

        loss_weights = self.class_loss_weights
        if loss_weights is not None:
            loss_weights = torch.tensor(loss_weights).float()
            args['weight'] = loss_weights

        ignore_class_index = self.ignore_class_index
        if ignore_class_index is not None:
            if ignore_class_index >= 0:
                args['ignore_index'] = ignore_class_index
            else:
                args['ignore_index'] = num_classes + ignore_class_index

        loss = nn.CrossEntropyLoss(**args)

        return loss

    def build_optimizer(self, model: nn.Module, **kwargs) -> optim.Adam:
        """Build and return an Adam optimizer for the given model.

        Args:
            model (nn.Module): Model to be trained.
            **kwargs: Extra args for the optimizer constructor.

        Returns:
            An Adam optimizer instance.
        """
        return optim.Adam(model.parameters(), lr=self.lr, **kwargs)

    def build_step_scheduler(self,
                             optimizer: optim.Optimizer,
                             train_ds_sz: int,
                             last_epoch: int = -1,
                             **kwargs) -> _LRScheduler | None:
        """Returns an LR scheduler that changes the LR each step.

        This is used to implement the "one cycle" schedule popularized by
        FastAI.

        Args:
            optimizer (optim.Optimizer): Optimizer to build scheduler for.
            train_ds_sz (int): Size of the training dataset.
            last_epoch (int): Last epoch. Defaults to -1.
            **kwargs: Extra args for the scheduler constructor.

        Returns:
            A step scheduler, if applicable. Otherwise, None.
        """
        scheduler = None
        if self.one_cycle and self.num_epochs > 1:
            steps_per_epoch = max(1, train_ds_sz // self.batch_sz)
            total_steps = self.num_epochs * steps_per_epoch
            step_size_up = (self.num_epochs // 2) * steps_per_epoch
            step_size_down = total_steps - step_size_up
            # Note that we don't pass in last_epoch here. See note below.
            scheduler = CyclicLR(
                optimizer,
                base_lr=self.lr / 10,
                max_lr=self.lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                cycle_momentum=kwargs.pop('cycle_momentum', False),
                **kwargs)
            # Note: We need this loop because trying to resume the scheduler by
            # just passing last_epoch does not work. See:
            # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822/2 # noqa
            num_past_epochs = last_epoch + 1
            for _ in range(num_past_epochs * steps_per_epoch):
                scheduler.step()
        return scheduler

    def build_epoch_scheduler(self,
                              optimizer: optim.Optimizer,
                              last_epoch: int = -1,
                              **kwargs) -> _LRScheduler | None:
        """Returns an LR scheduler that changes the LR each epoch.

        This is used to divide the learning rate by 10 at certain epochs.

        Args:
            optimizer (optim.Optimizer): Optimizer to build scheduler for.
            last_epoch (int): Last epoch. Defaults to -1.
            **kwargs: Extra args for the scheduler constructor.

        Returns:
            An epoch scheduler, if applicable. Otherwise, None.
        """
        scheduler = None
        if self.multi_stage:
            # Note that we don't pass in last_epoch here. See note below.
            scheduler = MultiStepLR(
                optimizer,
                milestones=self.multi_stage,
                gamma=kwargs.pop('gamma', 0.1),
                **kwargs)
            # Note: We need this loop because trying to resume the scheduler by
            # just passing last_epoch does not work. See:
            # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822/2 # noqa
            num_past_epochs = last_epoch + 1
            for _ in range(num_past_epochs):
                scheduler.step()
        return scheduler


def get_default_channel_display_groups(
        nb_img_channels: int) -> dict[str, ChannelInds]:
    """Returns the default channel_display_groups object.

    See PlotOptions.channel_display_groups.
    Displays at most the first 3 channels as RGB.

    Args:
        nb_img_channels: number of channels in the image that this is for
    """
    num_display_channels = min(3, nb_img_channels)
    return {'Input': list(range(num_display_channels))}


def validate_channel_display_groups(
        groups: dict[str, ChannelInds] | Sequence[ChannelInds] | None):
    """Validate channel display groups object.

    See PlotOptions.channel_display_groups.
    """
    if groups is None:
        return None
    elif len(groups) == 0:
        raise ConfigError(
            'channel_display_groups cannot be empty. Set to None instead.')
    elif not isinstance(groups, dict):
        # if in list/tuple form, convert to dict s.t.
        # [(0, 1, 2), (4, 3, 5)] --> {
        #   "Channels [0, 1, 2]": [0, 1, 2],
        #   "Channels [4, 3, 5]": [4, 3, 5]
        # }
        groups = {f'Channels: {[*chs]}': list(chs) for chs in groups}
    else:
        groups = {k: list(v) for k, v in groups.items()}

    if isinstance(groups, dict):
        for k, _v in groups.items():
            if not (0 < len(_v) <= 3):
                raise ConfigError(f'channel_display_groups[{k}]: '
                                  'len(group) must be 1, 2, or 3')
    return groups


@register_config('plot_options')
class PlotOptions(Config):
    """Config related to plotting."""
    transform: dict | None = Field(
        A.to_dict(MinMaxNormalize()),
        description='An Albumentations transform serialized as a dict that '
        'will be applied to each image before it is plotted. Mainly useful '
        'for undoing any data transformation that you do not want included in '
        'the plot, such as normalization. The default value will shift and scale the '
        'image so the values range from 0.0 to 1.0 which is the expected range for '
        'the plotting function. This default is useful for cases where the values after '
        'normalization are close to zero which makes the plot difficult to see.'
    )
    channel_display_groups: (
        dict[str, ChannelInds] | Sequence[ChannelInds] | None
    ) = Field(
        None,
        description=(
            'Groups of image channels to display together as a subplot '
            'when plotting the data and predictions. '
            'Can be a list or tuple of groups (e.g. [(0, 1, 2), (3,)]) or a '
            'dict containing title-to-group mappings '
            '(e.g. {"RGB": [0, 1, 2], "IR": [3]}), '
            'where each group is a list or tuple of channel indices and '
            'title is a string that will be used as the title of the subplot '
            'for that group.'))

    # validators
    _tf = field_validator('transform')(validate_albumentation_transform)

    def update(self, **kwargs) -> None:
        super().update()
        img_channels: int | None = kwargs.get('img_channels')
        if self.channel_display_groups is None and img_channels is not None:
            self.channel_display_groups = get_default_channel_display_groups(
                img_channels)

    @field_validator('channel_display_groups')
    @classmethod
    def validate_channel_display_groups(
            cls, v: dict[str, ChannelInds] | Sequence[ChannelInds] | None
    ) -> dict[str, ChannelInds] | None:
        return validate_channel_display_groups(v)


def data_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 1:
        cfg_dict['type_hint'] = 'image_data'
    elif version == 2:
        cfg_dict['img_channels'] = cfg_dict.get('img_channels')
    elif version == 6:
        class_names = cfg_dict.pop('class_names', [])
        class_colors = cfg_dict.pop('class_colors', [])
        cfg_dict['class_config'] = ClassConfig(
            names=class_names, colors=class_colors)
    return cfg_dict


@register_config('data', upgrader=data_config_upgrader)
class DataConfig(Config):
    """Config related to dataset for training and testing."""
    class_config: ClassConfig | None = Field(None, description='Class config.')
    img_channels: PosInt | None = Field(
        None, description='The number of channels of the training images.')
    img_sz: PosInt = Field(
        256,
        description=
        ('Length of a side of each image in pixels. This is the size to transform '
         'it to during training, not the size in the raw dataset.'))
    train_sz: int | None = Field(
        None,
        description=
        ('If set, the number of training images to use. If fewer images exist, '
         'then an exception will be raised.'))
    train_sz_rel: float | None = Field(
        None, description='If set, the proportion of training images to use.')
    num_workers: int = Field(
        4,
        description='Number of workers to use when DataLoader makes batches.')
    augmentors: list[str] = Field(
        default_augmentors,
        description='Names of albumentations augmentors to use for training '
        f'batches. Choices include: {augmentors}. Alternatively, a custom '
        'transform can be provided via the aug_transform option.')
    base_transform: dict | None = Field(
        None,
        description='An Albumentations transform serialized as a dict that '
        'will be applied to all datasets: training, validation, and test. '
        'This transformation is in addition to the resizing due to img_sz. '
        'This is useful for, for example, applying the same normalization to '
        'all datasets.')
    aug_transform: dict | None = Field(
        None,
        description='An Albumentations transform serialized as a dict that '
        'will be applied as data augmentation to the training dataset. This '
        'transform is applied before base_transform. If provided, the '
        'augmentors option is ignored.')
    plot_options: PlotOptions | None = Field(
        PlotOptions(), description='Options to control plotting.')
    preview_batch_limit: int | None = Field(
        None,
        description=
        ('Optional limit on the number of items in the preview plots produced '
         'during training.'))

    @property
    def class_names(self):
        if self.class_config is None:
            return None
        return self.class_config.names

    @property
    def class_colors(self):
        if self.class_config is None:
            return None
        return self.class_config.colors

    @property
    def num_classes(self):
        return len(self.class_config)

    # validators
    _base_tf = field_validator('base_transform')(
        validate_albumentation_transform)
    _aug_tf = field_validator('aug_transform')(
        validate_albumentation_transform)

    @field_validator('augmentors')
    @classmethod
    def validate_augmentors(cls, v: list[str]) -> list[str]:
        for aug_name in v:
            if aug_name not in augmentors:
                raise ConfigError(f'Unsupported augmentor "{aug_name}"')
        return v

    @model_validator(mode='after')
    def validate_plot_options(self) -> 'Self':
        if self.plot_options is not None and self.img_channels is not None:
            self.plot_options.update(img_channels=self.img_channels)
        return self

    def get_custom_albumentations_transforms(self) -> list[dict]:
        """Returns all custom transforms found in this config.

        This should return all serialized albumentations transforms with
        a 'lambda_transforms_path' field contained in this
        config or in any of its members no matter how deeply neseted.

        The purpose is to make it easier to adjust their paths all at once while
        saving to or loading from a bundle.
        """
        transforms_all = [
            self.base_transform, self.aug_transform,
            self.plot_options.transform
        ]
        transforms_with_lambdas = [
            tf for tf in transforms_all if (tf is not None) and (
                tf.get('lambda_transforms_path') is not None)
        ]
        return transforms_with_lambdas

    def get_bbox_params(self) -> A.BboxParams | None:
        """Returns BboxParams used by albumentations for data augmentation."""
        return None

    def get_data_transforms(self) -> tuple[A.BasicTransform, A.BasicTransform]:
        """Get albumentations transform objects for data augmentation.

        Returns a 2-tuple of a "base" transform and an augmentation transform.
        The base transform comprises a resize transform based on img_sz
        followed by the transform specified in base_transform. The augmentation
        transform comprises the base transform followed by either the transform
        in aug_transform (if specified) or the transforms in the augmentors
        field.

        The augmentation transform is intended to be used for training data,
        and the base transform for all other data where data augmentation is
        not desirable, such as validation or prediction.

        Returns:
            base transform and augmentation transform.
        """
        bbox_params = self.get_bbox_params()
        base_tfs = [A.Resize(self.img_sz, self.img_sz)]
        if self.base_transform is not None:
            base_tfs.append(
                deserialize_albumentation_transform(self.base_transform))
        base_transform = A.Compose(base_tfs, bbox_params=bbox_params)

        if self.aug_transform is not None:
            aug_transform = deserialize_albumentation_transform(
                self.aug_transform)
            aug_transform = A.Compose(
                [base_transform, aug_transform], bbox_params=bbox_params)
            return base_transform, aug_transform

        augmentors_dict = {
            'Blur': A.Blur(),
            'RandomRotate90': A.RandomRotate90(),
            'HorizontalFlip': A.HorizontalFlip(),
            'VerticalFlip': A.VerticalFlip(),
            'GaussianBlur': A.GaussianBlur(),
            'GaussNoise': A.GaussNoise(),
            'RGBShift': A.RGBShift(),
            'ToGray': A.ToGray()
        }
        aug_transforms = [base_transform]
        for augmentor in self.augmentors:
            try:
                aug_transforms.append(augmentors_dict[augmentor])
            except KeyError as k:
                log.warning(
                    f'{k} is an unknown augmentor. Continuing without {k}. '
                    f'Known augmentors are: {list(augmentors_dict.keys())}')
        aug_transform = A.Compose(aug_transforms, bbox_params=bbox_params)

        return base_transform, aug_transform

    def build(self,
              tmp_dir: str | None = None) -> tuple[Dataset, Dataset, Dataset]:
        """Build and return train, val, and test datasets."""
        raise NotImplementedError()

    def build_dataset(self,
                      split: Literal['train', 'valid', 'test'],
                      tmp_dir: str | None = None) -> Dataset:
        """Build and return dataset for a single split."""
        raise NotImplementedError()

    def random_subset_dataset(self,
                              ds: Dataset,
                              size: int | None = None,
                              fraction: Proportion | None = None) -> Subset:
        if size is None and fraction is None:
            return ds
        if size is not None and fraction is not None:
            raise ValueError('Specify either size or fraction but not both.')
        if fraction is not None:
            size = int(len(ds) * fraction)

        random.seed(1234)
        inds = list(range(len(ds)))
        random.shuffle(inds)
        ds = Subset(ds, inds[:size])
        return ds


@register_config('image_data')
class ImageDataConfig(DataConfig):
    """Config related to dataset for training and testing."""
    data_format: str | None = Field(
        None, description='Name of dataset format.')
    uri: str | list[str] | None = Field(
        None,
        description='One of the following:\n'
        '(1) a URI of a directory containing "train", "valid", and '
        '(optionally) "test" subdirectories;\n'
        '(2) a URI of a zip file containing (1);\n'
        '(3) a list of (2);\n'
        '(4) a URI of a directory containing zip files containing (1).')
    group_uris: list[str | list[str]] | None = Field(
        None,
        description=
        'This can be set instead of uri in order to specify groups of chips. '
        'Each element in the list is expected to be an object of the same '
        'form accepted by the uri field. The purpose of separating chips into '
        'groups is to be able to use the group_train_sz field.')
    group_train_sz: int | list[int] | None = Field(
        None,
        description='If group_uris is set, this can be used to specify the '
        'number of chips to use per group. Only applies to training chips. '
        'This can either be a single value that will be used for all groups '
        'or a list of values (one for each group).')
    group_train_sz_rel: Proportion | list[Proportion] | None = Field(
        None,
        description='Relative version of group_train_sz. Must be a float '
        'in [0, 1]. If group_uris is set, this can be used to specify the '
        'proportion of the total chips in each group to use per group. '
        'Only applies to training chips. This can either be a single value '
        'that will be used for all groups or a list of values '
        '(one for each group).')

    @model_validator(mode='after')
    def validate_group_uris(self) -> 'Self':
        group_train_sz = self.group_train_sz
        group_train_sz_rel = self.group_train_sz_rel
        group_uris = self.group_uris

        has_group_train_sz = group_train_sz is not None
        has_group_train_sz_rel = group_train_sz_rel is not None
        has_group_uris = group_uris is not None

        if has_group_train_sz and has_group_train_sz_rel:
            raise ConfigError('Only one of group_train_sz and '
                              'group_train_sz_rel should be specified.')
        if has_group_train_sz and not has_group_uris:
            raise ConfigError('group_train_sz specified without group_uris.')
        if has_group_train_sz_rel and not has_group_uris:
            raise ConfigError(
                'group_train_sz_rel specified without group_uris.')
        if has_group_train_sz and sequence_like(group_train_sz):
            if len(group_train_sz) != len(group_uris):
                raise ConfigError('len(group_train_sz) != len(group_uris).')
        if has_group_train_sz_rel and sequence_like(group_train_sz_rel):
            if len(group_train_sz_rel) != len(group_uris):
                raise ConfigError(
                    'len(group_train_sz_rel) != len(group_uris).')
        return self

    def _build_dataset(self,
                       dirs: Iterable[str],
                       tf: A.BasicTransform | None = None) -> Dataset:
        """Make datasets for a single split.

        Args:
            dirs: Directories where the data is located.
            tf: Transform for the dataset. Defaults to None.

        Returns:
            PyTorch-compatiable dataset.
        """
        per_dir_datasets = [self.dir_to_dataset(d, tf) for d in dirs]
        if len(per_dir_datasets) == 0:
            per_dir_datasets.append([])
        combined_dataset = ConcatDataset(per_dir_datasets)
        return combined_dataset

    def _build_datasets(self,
                        train_dirs: Iterable[str],
                        val_dirs: Iterable[str],
                        test_dirs: Iterable[str],
                        train_tf: A.BasicTransform | None = None,
                        val_tf: A.BasicTransform | None = None,
                        test_tf: A.BasicTransform | None = None
                        ) -> tuple[Dataset, Dataset, Dataset]:
        """Make training, validation, and test datasets.

        Args:
            train_dirs (str): Directories where training data is located.
            val_dirs (str): Directories where validation data is located.
            test_dirs (str): Directories where test data is located.
            train_tf (A.BasicTransform | None): Transform for the
                training dataset. Defaults to None.
            val_tf (A.BasicTransform | None): Transform for the
                validation dataset. Defaults to None.
            test_tf (A.BasicTransform | None): Transform for the
                test dataset. Defaults to None.

        Returns:
            PyTorch-compatiable training, validation, and test datasets.
        """
        train_ds = self._build_dataset(train_dirs, train_tf)
        val_ds = self._build_dataset(val_dirs, val_tf)
        test_ds = self._build_dataset(test_dirs, test_tf)
        return train_ds, val_ds, test_ds

    def dir_to_dataset(self, data_dir: str,
                       transform: A.BasicTransform) -> Dataset:
        raise NotImplementedError()

    def build(self, tmp_dir: str) -> tuple[Dataset, Dataset, Dataset]:

        if self.group_uris is None:
            return self._get_datasets_from_uri(self.uri, tmp_dir=tmp_dir)

        if self.uri is not None:
            log.warning('Both DataConfig.uri and DataConfig.group_uris '
                        'specified. Only DataConfig.group_uris will be used.')

        train_ds, valid_ds, test_ds = self._get_datasets_from_group_uris(
            self.group_uris, tmp_dir=tmp_dir)

        if self.train_sz is not None or self.train_sz_rel is not None:
            train_ds = self.random_subset_dataset(
                train_ds, size=self.train_sz, fraction=self.train_sz_rel)

        return train_ds, valid_ds, test_ds

    def build_dataset(self,
                      split: Literal['train', 'valid', 'test'],
                      tmp_dir: str | None = None) -> Dataset:

        if self.group_uris is None:
            ds = self._get_dataset_from_uri(
                self.uri, split=split, tmp_dir=tmp_dir)
            return ds

        if self.uri is not None:
            log.warning('Both DataConfig.uri and DataConfig.group_uris '
                        'specified. Only DataConfig.group_uris will be used.')

        ds = self._get_dataset_from_group_uris(
            self.group_uris, split=split, tmp_dir=tmp_dir)

        if split == 'train':
            if self.train_sz is not None or self.train_sz_rel is not None:
                ds = self.random_subset_dataset(
                    ds, size=self.train_sz, fraction=self.train_sz_rel)

        return ds

    def _get_datasets_from_uri(self, uri: str | list[str], tmp_dir: str
                               ) -> tuple[Dataset, Dataset, Dataset]:
        """Get image train, validation, & test datasets from a single zip file.

        Args:
            uri (str | list[str]): Uri of a zip file containing the
                images.

        Returns:
            Training, validation, and test dataSets.
        """
        data_dirs = self.get_data_dirs(uri, unzip_dir=tmp_dir)

        train_dirs = [join(d, 'train') for d in data_dirs if isdir(d)]
        val_dirs = [join(d, 'valid') for d in data_dirs if isdir(d)]
        test_dirs = [join(d, 'test') for d in data_dirs if isdir(d)]

        train_dirs = [d for d in train_dirs if isdir(d)]
        val_dirs = [d for d in val_dirs if isdir(d)]
        test_dirs = [d for d in test_dirs if isdir(d)]

        base_transform, aug_transform = self.get_data_transforms()
        train_tf = aug_transform
        val_tf, test_tf = base_transform, base_transform

        train_ds, val_ds, test_ds = self._build_datasets(
            train_dirs=train_dirs,
            val_dirs=val_dirs,
            test_dirs=test_dirs,
            train_tf=train_tf,
            val_tf=val_tf,
            test_tf=test_tf)
        return train_ds, val_ds, test_ds

    def _get_dataset_from_uri(self, uri: str | list[str],
                              split: Literal['train', 'valid', 'test'],
                              tmp_dir: str) -> Dataset:
        """Get image dataset from a single zip file.

        Args:
            uri: Uri of a zip file containing the images.

        Returns:
            Training, validation, and test dataSets.
        """
        data_dirs = self.get_data_dirs(uri, unzip_dir=tmp_dir)

        dirs = [join(d, split) for d in data_dirs if isdir(d)]
        dirs = [d for d in dirs if isdir(d)]

        base_transform, aug_transform = self.get_data_transforms()
        if split == 'train':
            tf = aug_transform
        else:
            tf = base_transform

        ds = self._build_dataset(dirs, tf)
        return ds

    def _get_datasets_from_group_uris(self,
                                      uris: str | list[str],
                                      tmp_dir: str,
                                      group_train_sz: int | None = None,
                                      group_train_sz_rel: float | None = None
                                      ) -> tuple[Dataset, Dataset, Dataset]:
        train_ds_lst, valid_ds_lst, test_ds_lst = [], [], []

        group_sizes = None
        if group_train_sz is not None:
            group_sizes = group_train_sz
        elif group_train_sz_rel is not None:
            group_sizes = group_train_sz_rel
        if not sequence_like(group_sizes):
            group_sizes = [group_sizes] * len(uris)

        for uri, size in zip(uris, group_sizes):
            train_ds, valid_ds, test_ds = self._get_datasets_from_uri(
                uri, tmp_dir=tmp_dir)
            if size is not None:
                if isinstance(size, float):
                    train_ds = self.random_subset_dataset(
                        train_ds, fraction=size)
                else:
                    train_ds = self.random_subset_dataset(train_ds, size=size)

            train_ds_lst.append(train_ds)
            valid_ds_lst.append(valid_ds)
            test_ds_lst.append(test_ds)

        train_ds, valid_ds, test_ds = (ConcatDataset(train_ds_lst),
                                       ConcatDataset(valid_ds_lst),
                                       ConcatDataset(test_ds_lst))
        return train_ds, valid_ds, test_ds

    def _get_dataset_from_group_uris(
            self,
            split: Literal['train', 'valid', 'test'],
            uris: str | list[str],
            tmp_dir: str,
            group_sz: int | None = None,
            group_sz_rel: float | None = None) -> Dataset:

        group_sizes = None
        if group_sz is not None:
            group_sizes = group_sz
        elif group_sz_rel is not None:
            group_sizes = group_sz_rel
        if not sequence_like(group_sizes):
            group_sizes = [group_sizes] * len(uris)

        per_uri_dataset = []
        for uri, size in zip(uris, group_sizes):
            ds = self._get_dataset_from_uri(uri, split=split, tmp_dir=tmp_dir)
            if size is not None:
                if isinstance(size, float):
                    ds = self.random_subset_dataset(ds, fraction=size)
                else:
                    ds = self.random_subset_dataset(ds, size=size)
            per_uri_dataset.append(ds)

        combined_dataset = ConcatDataset(per_uri_dataset)
        return combined_dataset

    def get_data_dirs(self, uri: str | list[str], unzip_dir: str) -> list[str]:
        """Extract data dirs from uri.

        Data dirs are directories containing  "train", "valid", and
        (optionally) "test" subdirectories.

        Args:
            uri: A URI or a list of URIs of one of the following:

                (1) a URI of a directory containing "train", "valid", and
                    (optionally) "test" subdirectories
                (2) a URI of a zip file containing (1)
                (3) a list of (2)
                (4) a URI of a directory containing zip files containing (1)
            unzip_dir (str): Directory where zip files will be extracted to, if
                needed.

        Returns:
            list[str]: Paths to directories that each contain contents of one
            zip file.
        """

        def is_data_dir(uri: str) -> bool:
            if not file_exists(uri, include_dir=True):
                return False
            paths = list_paths(uri)
            has_train = join(uri, 'train') in paths
            has_val = join(uri, 'valid') in paths
            return (has_train and has_val)

        if isinstance(uri, list):
            zip_uris = uri
            if not all(uri.endswith('.zip') for uri in zip_uris):
                raise ValueError('If uri is a list, all items must be URIs of '
                                 'zip files.')
        else:
            # if file
            if file_exists(uri, include_dir=False):
                if not uri.endswith('.zip'):
                    raise ValueError(
                        'URI is neither a directory nor a zip file.')
                zip_uris = [uri]
            # if dir
            elif file_exists(uri, include_dir=True):
                if is_data_dir(uri):
                    local_path = get_local_path(uri, unzip_dir)
                    if uri != local_path:
                        sync_from_dir(uri, local_path)
                    return [local_path]
                else:
                    zip_uris = list_paths(uri, ext='zip')
            # if non-existent
            else:
                raise FileNotFoundError(uri)

        data_dirs = self.unzip_data(zip_uris, unzip_dir)
        return data_dirs

    def unzip_data(self, zip_uris: list[str], unzip_dir: str) -> list[str]:
        """Unzip dataset zip files.

        Args:
            zip_uris (list[str]): A list of URIs of zip files:
            unzip_dir (str): Directory where zip files will be extracted to.

        Returns:
            list[str]: Paths to directories that each contain contents of one
            zip file.
        """
        data_dirs = []

        unzip_dir = join(unzip_dir, 'data', str(uuid.uuid4()))
        for i, zip_uri in enumerate(zip_uris):
            zip_path = download_if_needed(zip_uri)
            data_dir = join(unzip_dir, str(i))
            data_dirs.append(data_dir)
            unzip(zip_path, data_dir)

        return data_dirs


def geo_data_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 5:
        cfg_dict['sampling'] = cfg_dict.pop('window_opts', {})
    return cfg_dict


@register_config('geo_data', upgrader=geo_data_config_upgrader)
class GeoDataConfig(DataConfig):
    """Configure :class:`GeoDatasets <.GeoDataset>`.

    See :mod:`rastervision.pytorch_learner.dataset.dataset`.
    """

    scene_dataset: 'SceneDatasetConfig | None' = Field(None, description='')
    sampling: WindowSamplingConfig | dict[str, WindowSamplingConfig] = Field(
        {}, description='Window sampling config.')

    def __repr_args__(self):
        ds_str = repr(self.scene_dataset)
        if isinstance(self.sampling, dict):
            sampling_str = f'Dict with {len(self.sampling)} keys'
        else:
            sampling_str = str(self.sampling)
        out = [('scene_dataset', ds_str), ('sampling', sampling_str)]
        return out

    @model_validator(mode='after')
    def validate_sampling(self) -> 'Self':
        if not isinstance(self.sampling, dict):
            return self

        # empty dict
        if len(self.sampling) == 0:
            return self

        if self.scene_dataset is None:
            raise ConfigError('sampling is a non-empty dict but '
                              'scene_dataset is None.')

        for s in self.scene_dataset.all_scenes:
            if s.id not in self.sampling:
                raise ConfigError(
                    f'Window sampling config not found for scene: {s.id}')

        return self

    @model_validator(mode='after')
    def get_class_config_from_dataset_if_needed(self) -> 'Self':
        if self.class_config is None and self.scene_dataset is not None:
            self.class_config = self.scene_dataset.class_config
        return self

    def build_scenes(self,
                     scene_configs: Iterable['SceneConfig'],
                     tmp_dir: str | None = None) -> list[Scene]:
        """Build training, validation, and test scenes."""
        class_config = self.scene_dataset.class_config
        scenes = [
            s.build(class_config, tmp_dir, use_transformers=True)
            for s in scene_configs
        ]
        return scenes

    def _build_dataset(self,
                       split: Literal['train', 'valid', 'test'],
                       tf: A.BasicTransform | None = None,
                       tmp_dir: str | None = None,
                       **kwargs) -> Dataset:
        """Make training, validation, and test datasets.

        Args:
            split: Name of data split. One of: 'train', 'valid', 'test'.
            tf: Transform for the
                dataset. Defaults to None.
            tmp_dir: Temporary directory to be used for building scenes.
            **kwargs: Kwargs to pass to :meth:`.scene_to_dataset`.

        Returns:
            Dataset: PyTorch-compatiable dataset.
        """
        if self.scene_dataset is None:
            raise ValueError('Cannot build scenes if scene_dataset is None.')

        if split == 'train':
            scene_configs = self.scene_dataset.train_scenes
        elif split == 'valid':
            scene_configs = self.scene_dataset.validation_scenes
        elif split == 'test':
            scene_configs = self.scene_dataset.test_scenes

        scenes = self.build_scenes(scene_configs, tmp_dir)
        per_scene_datasets = [
            self.scene_to_dataset(s, tf, **kwargs) for s in scenes
        ]

        if len(per_scene_datasets) == 0:
            per_scene_datasets.append([])

        combined_dataset = ConcatDataset(per_scene_datasets)

        return combined_dataset

    def _build_datasets(self,
                        tmp_dir: str | None = None,
                        train_tf: A.BasicTransform | None = None,
                        val_tf: A.BasicTransform | None = None,
                        test_tf: A.BasicTransform | None = None,
                        **kwargs) -> tuple[Dataset, Dataset, Dataset]:
        """Make training, validation, and test datasets.

        Args:
            tmp_dir (str): Temporary directory to be used for building scenes.
            train_tf (A.BasicTransform | None): Transform for the
                training dataset. Defaults to None.
            val_tf (A.BasicTransform | None): Transform for the
                validation dataset. Defaults to None.
            test_tf (A.BasicTransform | None): Transform for the
                test dataset. Defaults to None.
            **kwargs: Kwargs to pass to :meth:`.scene_to_dataset`.

        Returns:
            tuple[Dataset, Dataset, Dataset]: PyTorch-compatiable training,
            validation, and test datasets.
        """
        train_ds = self._build_dataset('train', train_tf, tmp_dir, **kwargs)
        val_ds = self._build_dataset('valid', val_tf, tmp_dir, **kwargs)
        test_ds = self._build_dataset('test', test_tf, tmp_dir, **kwargs)
        return train_ds, val_ds, test_ds

    def scene_to_dataset(self,
                         scene: Scene,
                         transform: A.BasicTransform | None = None,
                         for_chipping: bool = False) -> Dataset:
        """Make a dataset from a single scene.
        """
        raise NotImplementedError()

    def build_dataset(self,
                      split: Literal['train', 'valid', 'test'],
                      tmp_dir: str | None = None) -> Dataset:

        base_transform, aug_transform = self.get_data_transforms()
        if split == 'train':
            tf = aug_transform
        else:
            tf = base_transform

        ds = self._build_dataset(split, tf, tmp_dir)

        if split == 'train':
            if self.train_sz is not None or self.train_sz_rel is not None:
                ds = self.random_subset_dataset(
                    ds, size=self.train_sz, fraction=self.train_sz_rel)

        return ds

    def build(self, tmp_dir: str | None = None,
              for_chipping: bool = False) -> tuple[Dataset, Dataset, Dataset]:
        base_transform, aug_transform = self.get_data_transforms()
        if for_chipping:
            train_tf, val_tf, test_tf = None, None, None
        else:
            train_tf = aug_transform
            val_tf, test_tf = base_transform, base_transform

        train_ds, val_ds, test_ds = self._build_datasets(
            tmp_dir=tmp_dir,
            train_tf=train_tf,
            val_tf=val_tf,
            test_tf=test_tf,
            for_chipping=for_chipping)

        if self.train_sz is not None or self.train_sz_rel is not None:
            train_ds = self.random_subset_dataset(
                train_ds, size=self.train_sz, fraction=self.train_sz_rel)

        return train_ds, val_ds, test_ds


def learner_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 4:
        # removed in version 5
        cfg_dict.pop('overfit_mode', None)
        cfg_dict.pop('test_mode', None)
        cfg_dict.pop('predict_mode', None)
    return cfg_dict


@register_config('learner', upgrader=learner_config_upgrader)
class LearnerConfig(Config):
    """Config for Learner."""
    model: ModelConfig | None = None
    solver: SolverConfig | None = None
    data: DataConfig

    eval_train: bool = Field(
        False,
        description='If True, runs final evaluation on training set '
        '(in addition to validation set). Useful for debugging.')
    save_model_bundle: bool = Field(
        True,
        description=
        ('If True, saves a model bundle at the end of training which '
         'is zip file with model and this LearnerConfig which can be used to make '
         'predictions on new images at a later time.'))
    log_tensorboard: bool = Field(
        True,
        description='Save Tensorboard log files at the end of each epoch.')
    run_tensorboard: bool = Field(
        False, description='run Tensorboard server during training')
    output_uri: str | None = Field(
        None, description='URI of where to save output')
    save_all_checkpoints: bool = Field(
        False,
        description=(
            'If True, all checkpoints would be saved. The latest checkpoint '
            'would be saved as `last-model.pth`. The checkpoints prior to '
            'last epoch are stored as `model-ckpt-epoch-{N}.pth` where `N` '
            'is the epoch number.'))

    @model_validator(mode='after')
    def validate_run_tensorboard(self) -> 'Self':
        if self.run_tensorboard and not self.log_tensorboard:
            raise ConfigError(
                'Cannot run tensorboard if log_tensorboard is False')
        return self

    @model_validator(mode='after')
    def validate_class_loss_weights(self) -> 'Self':
        if self.solver is None:
            return self
        class_loss_weights = self.solver.class_loss_weights
        if class_loss_weights is not None:
            num_weights = len(class_loss_weights)
            num_classes = self.data.num_classes
            if num_weights != num_classes:
                raise ConfigError(
                    f'class_loss_weights ({num_weights}) must be same length as '
                    f'the number of classes ({num_classes})')
        return self

    def build(self,
              tmp_dir: str | None = None,
              model_weights_path: str | None = None,
              model_def_path: str | None = None,
              loss_def_path: str | None = None,
              training: bool = True) -> 'Learner':
        """Returns a Learner instantiated using this Config.

        Args:
            tmp_dir (str): Root of temp dirs.
            model_weights_path (str): A local path to model weights.
                Defaults to None.
            model_def_path (str): A local path to a directory with a
                hubconf.py. If provided, the model definition is imported from
                here. Defaults to None.
            loss_def_path (str): A local path to a directory with a
                hubconf.py. If provided, the loss function definition is
                imported from here. Defaults to None.
            training (bool): Whether the model is to be used for
                training or prediction. If False, the model is put in eval mode
                and the loss function, optimizer, etc. are not initialized.
                Defaults to True.
        """
        raise NotImplementedError()

    def get_model_bundle_uri(self) -> str:
        """Returns the URI of where the model bundle is stored."""
        return join(self.output_uri, 'model-bundle.zip')
