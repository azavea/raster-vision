import rastervision as rv
from rastervision.task import TaskConfig
from rastervision.backend import BackendConfig


def check_analyzers_type(analyzers):
    if not isinstance(analyzers, list):
        raise rv.ConfigError(
            'analyzers must be a list of StatsAnalyzerConfig objects, got {}'.
            format(type(analyzers)))


def check_backend_type(backend):
    if not issubclass(type(backend), BackendConfig):
        raise rv.ConfigError(
            'Backend must be a child of class BackendConfig, got {}'.format(
                type(backend)))


def check_scenes_type(scenes):
    if not isinstance(scenes, list):
        raise rv.ConfigError(
            'scenes must be a list of SceneConfig objects, got {}'.format(
                type(scenes)))


def check_task_type(task):
    if not issubclass(type(task), TaskConfig):
        raise rv.ConfigError(
            'Task must be a child class of TaskConfig, got {}'.format(
                type(task)))
