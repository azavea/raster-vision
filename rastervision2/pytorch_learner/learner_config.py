from typing import List, Optional, Union, Tuple, TYPE_CHECKING
from os.path import join
import importlib
from enum import Enum

from pydantic import PositiveFloat, PositiveInt

from rastervision2.pipeline.config import (Config, register_config,
                                           ConfigError, Field)
from rastervision2.pytorch_learner.utils import color_to_triple

default_augmentors = ['RandomRotate90', 'HorizontalFlip', 'VerticalFlip']
augmentors = [
    'Blur', 'RandomRotate90', 'HorizontalFlip', 'VerticalFlip', 'GaussianBlur',
    'GaussNoise', 'RGBShift', 'ToGray'
]

if TYPE_CHECKING:
    from rastervision2.pytorch_learner.learner import Learner  # noqa


def get_torchvision_backbones():
    backbones = []
    # This may need to be updated after upgrading torchvision.
    packages = [
        'alexnet', 'densenet', 'googlenet', 'inception', 'mnasnet',
        'mobilenet', 'resnet', 'shufflenetv2', 'squeezenet', 'vgg'
    ]
    for package in packages:
        module = importlib.import_module(
            'torchvision.models.{}'.format(package))
        backbones.extend(module.__all__)

    return backbones


backbones = get_torchvision_backbones()


class Backbone(Enum):
    alexnet = 1
    densenet121 = 2
    densenet169 = 3
    densenet201 = 4
    densenet161 = 5
    googlenet = 6
    inception_v3 = 7
    mnasnet0_5 = 8
    mnasnet0_75 = 9
    mnasnet1_0 = 10
    mnasnet1_3 = 11
    mobilenet_v2 = 12
    resnet18 = 13
    resnet34 = 14
    resnet50 = 15
    resnet101 = 16
    resnet152 = 17
    resnext50_32x4d = 18
    resnext101_32x8d = 19
    wide_resnet50_2 = 20
    wide_resnet101_2 = 21
    shufflenet_v2_x0_5 = 22
    shufflenet_v2_x1_0 = 23
    shufflenet_v2_x1_5 = 24
    shufflenet_v2_x2_0 = 25
    squeezenet1_0 = 26
    squeezenet1_1 = 27
    vgg11 = 28
    vgg11_bn = 29
    vgg13 = 30
    vgg13_bn = 31
    vgg16 = 32
    vgg16_bn = 33
    vgg19_bn = 34
    vgg19 = 35


@register_config('model')
class ModelConfig(Config):
    """Config related to models."""
    backbone: Backbone = Field(
        Backbone.resnet18,
        description='The torchvision.models backbone to use.')
    pretrained: bool = Field(
        True, description=(
            'If True, use ImageNet weights. If False, use random initialization.'))
    init_weights: Optional[str] = Field(
        None,
        description=(
            'URI of PyTorch model weights used to initialize model. '
            'If set, this supercedes the pretrained option.'))

    def update(self, learner: Optional['LearnerConfig'] = None):
        pass

    def get_backbone_str(self):
        return self.backbone.name


@register_config('solver')
class SolverConfig(Config):
    """Config related to solver aka optimizer."""
    lr: PositiveFloat = Field(1e-4, description='Learning rate.')
    num_epochs: PositiveInt = Field(
        10,
        description=
        'Number of epochs (ie. sweeps through the whole training set).')
    test_num_epochs: PositiveInt = Field(
        2, description='Number of epochs to use in test mode.')
    test_batch_sz: PositiveInt = Field(
        4, description='Batch size to use in test mode.')
    overfit_num_steps: PositiveInt = Field(
        1, description='Number of optimizer steps to use in overfit mode.')
    sync_interval: PositiveInt = Field(
        1, description='The interval in epochs for each sync to the cloud.')
    batch_sz: PositiveInt = Field(32, description='Batch size.')
    one_cycle: bool = Field(
        True,
        description=
        ('If True, use triangular LR scheduler with a single cycle across all '
         'epochs with start and end LR being lr/10 and the peak being lr.'))
    multi_stage: List = Field(
        [], description=('List of epoch indices at which to divide LR by 10.'))

    def update(self, learner: Optional['LearnerConfig'] = None):
        pass


@register_config('data')
class DataConfig(Config):
    """Config related to dataset for training and testing."""
    # TODO shouldn't this be required?
    uri: Optional[str] = Field(
        None,
        description=
        ('URI of the dataset. This can be a zip file, or a directory which contains '
         'a set of zip files.'))
    data_format: Optional[str] = Field(
        None, description='Name of dataset format.')
    class_names: List[str] = Field([], description='Names of classes.')
    class_colors: Union[None, List[str], List[Tuple]] = Field(
        None, description='Colors used to display classes.')
    img_sz: PositiveInt = Field(
        256,
        description=
        ('Length of a side of each image in pixels. This is the size to transform '
         'it to during training, not the size in the raw dataset.'))
    num_workers: int = Field(
        4,
        description='Number of workers to use when DataLoader makes batches.')
    # TODO support setting parameters of augmentors?
    augmentors: List[str] = Field(
        default_augmentors,
        description=(
            'Names of albumentations augmentors to use for training batches. '
            'Choices include: ' + str(augmentors)))

    def update(self, learner: Optional['LearnerConfig'] = None):
        if not self.class_colors:
            self.class_colors = [color_to_triple() for _ in self.class_names]

    def validate_augmentors(self):
        self.validate_list('augmentors', augmentors)

    def validate_config(self):
        self.validate_augmentors()


@register_config('learner')
class LearnerConfig(Config):
    """Config for Learner."""
    model: ModelConfig
    solver: SolverConfig
    data: DataConfig

    predict_mode: bool = Field(
        False,
        description='If True, skips training, loads model, and does final eval.'
    )
    test_mode: bool = Field(
        False,
        description=
        ('If True, uses test_num_epochs, test_batch_sz, truncated datasets with '
         'only a single batch, image_sz that is cut in half, and num_workers = 0. '
         'This is useful for testing that code runs correctly on CPU without '
         'multithreading before running full job on GPU.'))
    overfit_mode: bool = Field(
        False,
        description=
        ('If True, uses half image size, and instead of doing epoch-based training, '
         'optimizes the model using a single batch repeatedly for '
         'overfit_num_steps number of steps.'))
    eval_train: bool = Field(
        False,
        description=
        ('If True, runs final evaluation on training set (in addition to test set). '
         'Useful for debugging.'))
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
    output_uri: Optional[str] = Field(
        None, description='URI of where to save output')

    def update(self):
        super().update()

        if self.overfit_mode:
            self.data.img_sz = self.data.img_sz // 2
            if self.test_mode:
                self.solver.overfit_num_steps = self.solver.test_overfit_num_steps

        if self.test_mode:
            self.solver.num_epochs = self.solver.test_num_epochs
            self.solver.batch_sz = self.solver.test_batch_sz
            self.data.img_sz = self.data.img_sz // 2
            self.data.num_workers = 0

        self.model.update(learner=self)
        self.solver.update(learner=self)
        self.data.update(learner=self)

    def validate_config(self):
        if self.run_tensorboard and not self.log_tensorboard:
            raise ConfigError(
                'Cannot run_tensorboard if log_tensorboard is False')

    def build(self, tmp_dir: str,
              model_path: Optional[str] = None) -> 'Learner':
        """Returns a Learner instantiated using this Config.

        Args:
            tmp_dir: root of temp dirs
            model_path: local path to model weights. If this is passed, the Learner
                is assumed to be used to make predictions and not train a model.
        """
        raise NotImplementedError()

    def get_model_bundle_uri(self) -> str:
        """Returns the URI of where the model bundel is stored."""
        return join(self.output_uri, 'model-bundle.zip')
