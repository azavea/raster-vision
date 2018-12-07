import rastervision as rv
from rastervision.task import TaskConfig
from rastervision.backend import BackendConfig
from rastervision.analyzer import AnalyzerConfig
from rastervision.data import SceneConfig


def check_analyzers_type(analyzers):
    if not isinstance(analyzers, list):
        raise rv.ConfigError(
            'analyzers must be a list of AnalyzerConfig objects, got {}'.
            format(type(analyzers)))
    for analyzer in analyzers:
        if not issubclass(type(analyzer), AnalyzerConfig):
            if not isinstance(analyzer, str):
                raise rv.ConfigError(
                    'analyzers must be of class AnalyzerConfig or string, got {}'.
                    format(type(analyzer)))


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
    for scene in scenes:
        if not isinstance(scene, SceneConfig):
            if not isinstance(scene, str):
                raise rv.ConfigError(
                    'scene must be a SceneConfig object or str, got {}'.format(
                        type(scene)))


def check_task_type(task):
    if not issubclass(type(task), TaskConfig):
        raise rv.ConfigError(
            'Task must be a child class of TaskConfig, got {}'.format(
                type(task)))
