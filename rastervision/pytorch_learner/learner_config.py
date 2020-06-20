from typing import List, Optional, Union, TYPE_CHECKING
from os.path import join
from enum import Enum

from pydantic import PositiveFloat, PositiveInt

from rastervision.pipeline.config import (Config, register_config, ConfigError,
                                          Field)
from rastervision.pytorch_learner.utils import color_to_triple

default_augmentors = ['RandomRotate90', 'HorizontalFlip', 'VerticalFlip']
augmentors = [
    'Blur', 'RandomRotate90', 'HorizontalFlip', 'VerticalFlip', 'GaussianBlur',
    'GaussNoise', 'RGBShift', 'ToGray'
]

if TYPE_CHECKING:
    from rastervision.pytorch_learner.learner import Learner  # noqa


class Backbone(Enum):
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
    init_weights: Optional[str] = Field(
        None,
        description=('URI of PyTorch model weights used to initialize model. '
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
    uri: Union[None, str, List[str]] = Field(
        None,
        description=
        ('URI of the dataset. This can be a zip file, a list of zip files, or a '
         'directory which contains a set of zip files.'))
    train_sz: Optional[int] = Field(
        None,
        description=
        ('If set, the number of training images to use. If fewer images exist, '
         'then an exception will be raised.'))
    group_uris: Union[None, List[Union[str, List[str]]]] = Field(
        None,
        description=
        ('This can be set instead of uri in order to specify groups of chips. Each '
         'element in the list is expected to be an object of the same form accepted by '
         'the uri field. The purpose of separating chips into groups is to be able to '
         'use the group_train_sz field.'))
    group_train_sz: Optional[int] = Field(
        None,
        description=
        ('If group_uris is set, this can be used to specify the number of chips to use '
         'per group.'))
    data_format: Optional[str] = Field(
        None, description='Name of dataset format.')
    class_names: List[str] = Field([], description='Names of classes.')
    class_colors: Union[None, List[str], List[List]] = Field(
        None,
        description=('Colors used to display classes. '
                     'Can be color 3-tuples in list form.'))
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
