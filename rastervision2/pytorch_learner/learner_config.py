from typing import List, Optional, TYPE_CHECKING
from os.path import join
import importlib

from rastervision2.pipeline.config import (Config, register_config,
                                           ConfigError, Field)

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


@register_config('model')
class ModelConfig(Config):
    """Config related to models."""
    backbone: str = Field(
        'resnet18', description='name of torchvision.models backbone to use')
    init_weights: Optional[str] = Field(None, description=(
        'URI of PyTorch model weights used to initialize model. If None, '
        'will use Imagenet pretrained model weights provided by torchvision.'))

    def update(self, learner: Optional['LearnerConfig'] = None):
        pass

    def validate_backbone(self):
        self.validate_list('backbone', backbones)

    def validate_config(self):
        self.validate_backbone()


@register_config('solver')
class SolverConfig(Config):
    """Config related to solver aka optimizer."""
    lr: float = Field(1e-4, description='Learning rate.')
    num_epochs: int = Field(
        10, description='Number of epochs (ie. sweeps through the whole training set).')
    test_num_epochs: int = Field(2, description='Number of epochs to use in test mode.')
    test_batch_sz: int = Field(4, description='Batch size to use in test mode.')
    overfit_num_steps: int = Field(
        1, description='Number of optimizer steps to use in overfit mode.')
    sync_interval: int = Field(
        1, description='The interval in epochs for each sync to the cloud.')
    batch_sz: int = Field(32, description='Batch size.')
    one_cycle: bool = Field(True, description=(
        'If True, use triangular LR scheduler with a single cycle across all '
        'epochs with start and end LR being lr/10 and the peak being lr.'))
    multi_stage: List = Field([], description=(
        'List of epoch indices at which to divide LR by 10.'))

    def update(self, learner: Optional['LearnerConfig'] = None):
        pass

    def validate_config(self):
        self.validate_nonneg('lr')
        self.validate_nonneg('num_epochs')
        self.validate_nonneg('test_num_epochs')
        self.validate_nonneg('overfit_num_steps')
        self.validate_nonneg('sync_interval')
        self.validate_nonneg('batch_sz')


@register_config('data')
class DataConfig(Config):
    """Config related to dataset for training and testing."""
    # TODO shouldn't this be required?
    uri: Optional[str] = Field(None, description=(
        'URI of the dataset. This can be a zip file, or a directory which contains '
        'a set of zip files.'))
    data_format: Optional[str] = Field(None, description='Name of dataset format.')
    class_names: List[str] = Field([], description='Names of classes.')
    # TODO make this optional
    class_colors: List[str] = Field([], description='Colors used to display classes.')
    img_sz: int = Field(256, description=(
        'Length of a side of each image in pixels. This is the size to transform '
        'it to during training, not the size in the raw dataset.'))
    num_workers: int = Field(
        4, description='Number of workers to use when DataLoader makes batches.')
    # TODO support setting parameters of augmentors?
    augmentors: List[str] = Field(default_augmentors, description=(
        'Names of albumentations augmentors to use for training batches. '
        'Choices include: ' + str(augmentors)))

    def update(self, learner: Optional['LearnerConfig'] = None):
        pass

    def validate_augmentors(self):
        self.validate_list('augmentors', augmentors)

    def validate_data_format(self):
        raise NotImplementedError()

    def validate_config(self):
        self.validate_nonneg('img_sz')
        self.validate_nonneg('num_workers')
        self.validate_augmentors()
        self.validate_data_format()


@register_config('learner')
class LearnerConfig(Config):
    """Config for Learner."""
    model: ModelConfig
    solver: SolverConfig
    data: DataConfig

    predict_mode: bool = Field(
        False, description='If True, skips training, loads model, and does final eval.')
    test_mode: bool = Field(
        False, description=(
            'If True, uses test_num_epochs, test_batch_sz, truncated datasets with '
            'only a single batch, image_sz that is cut in half, and num_workers = 0. '
            'This is useful for testing that code runs correctly on CPU without '
            'multithreading before running full job on GPU.'))
    overfit_mode: bool = Field(False, description=(
        'If True, uses half image size, and instead of doing epoch-based training, '
        'optimizes the model using a single batch repeatedly for '
        'overfit_num_steps number of steps.'))
    eval_train: bool = Field(False, description=(
        'If True, runs final evaluation on training set (in addition to test set). '
        'Useful for debugging.'))
    save_model_bundle: bool = Field(True, description=(
        'If True, saves a model bundle at the end of training which '
        'is zip file with model and this LearnerConfig which can be used to make '
        'predictions on new images at a later time.'))
    log_tensorboard: bool = Field(
        True, description='Save Tensorboard log files at the end of each epoch.')
    run_tensorboard: bool = Field(
        False, description='run Tensorboard server during training')
    output_uri: Optional[str] = Field(None, description='URI of where to save output')

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
