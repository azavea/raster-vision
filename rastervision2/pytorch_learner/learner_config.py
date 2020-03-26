from typing import List, Optional, TYPE_CHECKING
from os.path import join
import importlib

from rastervision2.pipeline.config import (Config, register_config,
                                           ConfigError)

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
    """Config related to models.

    Attributes:
        backbone: name of torchvision.models backbone to use
        init_weights: URI of PyTorch model weights used to initialize model. If None,
            will use Imagenet pretrained model weights provided by torchvision.
    """
    backbone: str = 'resnet18'
    init_weights: Optional[str] = None

    def update(self, learner: Optional['LearnerConfig'] = None):
        pass

    def validate_backbone(self):
        self.validate_list('backbone', backbones)

    def validate_config(self):
        self.validate_backbone()


@register_config('solver')
class SolverConfig(Config):
    """Config related to solver aka optimizer.

    Attributes:
        lr: learning rate
        num_epochs: number of epochs (ie. sweeps through the whole training set)
        test_num_epochs: number of epochs to use in test mode
        test_batch_sz: batch size to use in test mode
        overfit_num_steps: number of optimizer steps to use in overfit mode
        sync_interval: syncs output to cloud every sync_interval epochs
        batch_sz: batch size
        one_cycle: if True, use triangular LR scheduler with a single cycle across all
            epochs with start and end LR being lr/10 and the peak being lr
        multi_stage: list of epoch indices at which to divide LR by 10
    """
    lr: float = 1e-4
    num_epochs: int = 10
    test_num_epochs: int = 2
    test_batch_sz: int = 4
    overfit_num_steps: int = 1
    sync_interval: int = 1
    batch_sz: int = 32
    one_cycle: bool = True
    multi_stage: List = []

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
    """Config related to dataset.

    Attributes:
        uri: URI of the dataset. This can be a zip file, or a directory which contains
            a set of zip files.
        data_format: name of dataset format
        class_names: names of classes
        class_colors: colors used to display classes
        img_sz: length of a side of each image in pixels. This is the size to transform
            it to during training, not the size in the raw dataset.
        num_workers: number of workers to use when DataLoader makes batches
        augmentors: names of albumentations augmentors to use. Defaults to
            ['RandomRotate90', 'HorizontalFlip', 'VerticalFlip']. Other options include:
            ['Blur', 'RandomRotate90', 'HorizontalFlip', 'VerticalFlip', 'GaussianBlur',
            'GaussNoise', 'RGBShift', 'ToGray']
    """
    # TODO shouldn't this be required?
    uri: Optional[str] = None
    data_format: Optional[str] = None
    class_names: List[str] = []
    # TODO make this optional
    class_colors: List[str] = []
    img_sz: int = 256
    num_workers: int = 4
    # TODO support setting parameters of augmentors?
    augmentors: List[str] = default_augmentors

    def update(self, learner: Optional['LearnerConfig'] = None):
        pass

    def validate_augmentors(self):
        self.validate_list('augmentors', augmentors)

    def validate_data_format(self):
        raise NotImplementedError()

    def validate_config(self):
        if len(self.class_names) != len(self.class_colors):
            raise ConfigError('len(class_names) must equal len(class_colors')

        self.validate_nonneg('img_sz')
        self.validate_nonneg('num_workers')
        self.validate_augmentors()
        self.validate_data_format()


@register_config('learner')
class LearnerConfig(Config):
    """Config for Learner.

    Attribute:
        predict_mode: if True, skips training, loads model, and does final eval
        test_mode: if True, uses test_num_epochs, test_batch_sz, truncated datasets with
            only a single batch, image_sz that is cut in half, and num_workers = 0. This
            is useful for testing that code runs correctly on CPU without multithreading
            before running full job on GPU.
        overfit_mode: if True, uses half image size, and instead of doing epoch-based
            training, optimizes the model using a single batch repeatedly for
            overfit_num_steps number of steps.
        eval_train: if True, runs final evaluation on training set
            (in addition to test set). Useful for debugging.
        save_model_bundle: if True, saves a model bundle at the end of training which
            is zip file with model and this LearnerConfig which can be used to make
            predictions on new images at a later time.
        log_tensorboard: save Tensorboard log files at the end of each epoch
        run_tensorboard: run Tensorboard server during training
        output_uri: URI of where to save output
    """
    model: ModelConfig
    solver: SolverConfig
    data: DataConfig

    predict_mode: bool = False
    test_mode: bool = False
    overfit_mode: bool = False
    eval_train: bool = False
    save_model_bundle: bool = True
    log_tensorboard: bool = True
    run_tensorboard: bool = False
    output_uri: Optional[str] = None

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
