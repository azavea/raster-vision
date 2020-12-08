from typing import Union, Optional
from enum import Enum

import albumentations as A

from torch.utils.data import Dataset

from rastervision.core.data import Scene
from rastervision.pipeline.config import (Config, register_config, ConfigError)
from rastervision.pytorch_learner.learner_config import (
    LearnerConfig, ModelConfig, ImageDataConfig, GeoDataConfig,
    GeoDataWindowMethod)
from rastervision.pytorch_learner.image_folder import ImageFolder
from rastervision.pytorch_learner.dataset import (
    ClassificationImageDataset, ClassificationSlidingWindowGeoDataset,
    ClassificationRandomWindowGeoDataset)


class ClassificationDataFormat(Enum):
    image_folder = 'image_folder'


def clf_data_config_upgrader(cfg_dict, version):
    if version == 1:
        cfg_dict['type_hint'] = 'classification_image_data'
    return cfg_dict


@register_config('classification_data', upgrader=clf_data_config_upgrader)
class ClassificationDataConfig(Config):
    pass


@register_config('classification_image_data')
class ClassificationImageDataConfig(ClassificationDataConfig, ImageDataConfig):
    data_format: ClassificationDataFormat = ClassificationDataFormat.image_folder

    def dir_to_dataset(self, data_dir: str,
                       transform: A.BasicTransform) -> Dataset:
        ds = ClassificationImageDataset(
            ImageFolder(data_dir, classes=self.class_names),
            transform=transform)
        return ds


@register_config('classification_geo_data')
class ClassificationGeoDataConfig(ClassificationDataConfig, GeoDataConfig):
    def build_scenes(self, tmp_dir: str):
        for s in self.scene_dataset.train_scenes:
            s.label_source.lazy = True
        for s in self.scene_dataset.validation_scenes:
            s.label_source.lazy = True
        for s in self.scene_dataset.test_scenes:
            s.label_source.lazy = True
        return super().build_scenes(tmp_dir=tmp_dir)

    def scene_to_dataset(self,
                         scene: Scene,
                         transform: Optional[A.BasicTransform] = None
                         ) -> Dataset:
        if isinstance(self.window_opts, dict):
            opts = self.window_opts[scene.id]
        else:
            opts = self.window_opts

        if opts.method == GeoDataWindowMethod.sliding:
            ds = ClassificationSlidingWindowGeoDataset(
                scene,
                size=opts.size,
                stride=opts.stride,
                padding=opts.padding,
                transform=transform)
        elif opts.method == GeoDataWindowMethod.random:
            ds = ClassificationRandomWindowGeoDataset(
                scene,
                size_lims=opts.size_lims,
                h_lims=opts.h_lims,
                w_lims=opts.w_lims,
                out_size=opts.size,
                padding=opts.padding,
                max_windows=opts.max_windows,
                max_sample_attempts=opts.max_sample_attempts,
                transform=transform)
        else:
            raise NotImplementedError()
        return ds


@register_config('classification_model')
class ClassificationModelConfig(ModelConfig):
    pass


@register_config('classification_learner')
class ClassificationLearnerConfig(LearnerConfig):
    data: Union[ClassificationImageDataConfig, ClassificationGeoDataConfig]
    model: ClassificationModelConfig

    def build(self,
              tmp_dir,
              model_path=None,
              model_def_path=None,
              loss_def_path=None,
              training=True):
        from rastervision.pytorch_learner.classification_learner import (
            ClassificationLearner)
        return ClassificationLearner(
            self,
            tmp_dir=tmp_dir,
            model_path=model_path,
            model_def_path=model_def_path,
            loss_def_path=loss_def_path,
            training=training)

    def validate_config(self):
        super().validate_config()
        self.validate_class_loss_weights()
        self.validate_ignore_last_class()

    def validate_ignore_last_class(self):
        if self.solver.ignore_last_class:
            raise ConfigError(
                'ignore_last_class is not supported for Chip Classification.')

    def validate_class_loss_weights(self):
        if self.solver.class_loss_weights is None:
            return

        num_weights = len(self.solver.class_loss_weights)
        num_classes = len(self.data.class_names)
        if num_weights != num_classes:
            raise ConfigError(
                f'class_loss_weights ({num_weights}) must be same length as '
                f'the number of classes ({num_classes})')
