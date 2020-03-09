from typing import List
from os.path import join
import importlib

from rastervision2.pipeline.pipeline_config import PipelineConfig
from rastervision2.pipeline.config import (Config, register_config,
                                           ConfigError)

default_augmentors = ['RandomRotate90', 'HorizontalFlip', 'VerticalFlip']
augmentors = [
    'Blur', 'RandomRotate90', 'HorizontalFlip', 'VerticalFlip', 'GaussianBlur',
    'GaussNoise', 'RGBShift', 'ToGray'
]


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
    backbone: str = 'resnet18'
    init_weights: str = None

    def update(self, learner=None):
        pass

    def validate_backbone(self):
        self.validate_list('backbone', backbones)

    def validate_config(self):
        self.validate_backbone()


@register_config('solver')
class SolverConfig(Config):
    lr: float = 1e-4
    num_epochs: int = 10
    test_num_epochs: int = 2
    overfit_num_steps: int = 1
    sync_interval: int = 1
    batch_sz: int = 32
    one_cycle: bool = True
    multi_stage: List = []

    def update(self, learner=None):
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
    uri: str = None
    data_format: str = None
    class_names: List[str] = []
    class_colors: List[str] = []
    img_sz: int = 256
    num_workers: int = 4
    augmentors: List[str] = default_augmentors

    def update(self, learner=None):
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
    output_uri: str = None

    def update(self):
        super().update()

        if self.overfit_mode:
            self.data.img_sz = self.data.img_sz // 2
            if self.test_mode:
                self.solver.overfit_num_steps = self.solver.test_overfit_num_steps

        if self.test_mode:
            self.solver.num_epochs = self.solver.test_num_epochs
            self.data.img_sz = self.data.img_sz // 2
            self.solver.batch_sz = 4
            self.data.num_workers = 0

        self.model.update(learner=self)
        self.solver.update(learner=self)
        self.data.update(learner=self)

    def validate_config(self):
        if self.run_tensorboard and not self.log_tensorboard:
            raise ConfigError(
                'Cannot run_tensorboard if log_tensorboard is False')

    def build(self, tmp_dir, model_path=None):
        raise NotImplementedError()

    def get_model_bundle_uri(self):
        return join(self.output_uri, 'model-bundle.zip')

    def build_from_model_bundle(self, model_bundle_path, tmp_dir):
        from rastervision2.pytorch_learner import Learner
        return Learner.from_model_bundle(model_bundle_path, tmp_dir)


@register_config('learner_pipeline')
class LearnerPipelineConfig(PipelineConfig):
    learner: LearnerConfig

    def update(self):
        super().update()

        if self.learner.output_uri is None:
            self.learner.output_uri = self.root_uri

        self.learner.update()

    def build(self, tmp_dir):
        from rastervision2.pytorch_learner.learner_pipeline import LearnerPipeline
        return LearnerPipeline(self, tmp_dir)
