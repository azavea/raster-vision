from rastervision.pipeline.config import (register_config, validator,
                                          ConfigError)
from rastervision.pytorch_backend.pytorch_learner_backend_config import (
    PyTorchLearnerBackendConfig)
from rastervision.pytorch_learner.learner_config import default_augmentors
from rastervision.pytorch_learner.object_detection_learner_config import (
    ObjectDetectionModelConfig, ObjectDetectionLearnerConfig,
    ObjectDetectionImageDataConfig)
from rastervision.pytorch_backend.pytorch_object_detection import (
    PyTorchObjectDetection)


def objdet_learner_backend_config_upgrader(cfg_dict, version):
    if version == 0:
        fields = {
            'augmentors': default_augmentors,
            'group_uris': None,
            'group_train_sz': None,
            'group_train_sz_rel': None,
            'num_workers': 4,
            'img_sz': None,
            'base_transform': None,
            'aug_transform': None,
            'plot_options': None,
            'preview_batch_limit': None
        }
        data_cfg_dict = {
            key: cfg_dict.pop(key, default_val)
            for key, default_val in fields.items() if key in cfg_dict
        }
        if data_cfg_dict['img_sz'] is None:
            data_cfg_dict['img_sz'] = 256

        data_cfg = ObjectDetectionImageDataConfig(**data_cfg_dict)
        data_cfg.update()
        data_cfg.validate_config()
        cfg_dict['data'] = data_cfg.dict()
    return cfg_dict


@register_config(
    'pytorch_object_detection_backend',
    upgrader=objdet_learner_backend_config_upgrader)
class PyTorchObjectDetectionConfig(PyTorchLearnerBackendConfig):
    model: ObjectDetectionModelConfig

    def get_learner_config(self, pipeline):
        learner = ObjectDetectionLearnerConfig(
            data=self.data,
            model=self.model,
            solver=self.solver,
            test_mode=self.test_mode,
            output_uri=pipeline.train_uri,
            log_tensorboard=self.log_tensorboard,
            run_tensorboard=self.run_tensorboard)
        learner.update()
        return learner

    def build(self, pipeline, tmp_dir):
        learner = self.get_learner_config(pipeline)
        return PyTorchObjectDetection(pipeline, learner, tmp_dir)

    @validator('model')
    def validate_model_config(cls, v):
        if v.external_def is not None:
            raise ConfigError('external_def is currently not supported for '
                              'Object Detection.')
        return v

    @validator('solver')
    def validate_solver_config(cls, v):
        if v.ignore_last_class:
            raise ConfigError(
                'ignore_last_class is not supported for Object Detection.')
        if v.class_loss_weights is not None:
            raise ConfigError(
                'class_loss_weights is currently not supported for '
                'Object Detection.')
        if v.external_loss_def is not None:
            raise ConfigError(
                'external_loss_def is currently not supported for '
                'Object Detection.')
        return v
