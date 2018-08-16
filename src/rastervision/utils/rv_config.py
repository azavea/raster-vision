import configparser
import os
from os.path import join
from pathlib import Path

from rastervision.utils.files import (file_to_str, NotReadableError)

HOME_CONFIG_PATH = join(str(Path.home()), '.rastervision', 'config.ini')


def _get_profile(profile=None):
    if profile is not None:
        return profile
    return os.environ.get('RV_PROFILE', 'default')


def _get_from_args(batch_job_queue=None, batch_job_def=None, github_repo=None):
    return {
        'batch_job_def': batch_job_def,
        'batch_job_queue': batch_job_queue,
        'github_repo': github_repo
    }


def _get_empty():
    return _get_from_args()


def _get_from_env():
    return {
        'batch_job_def': os.environ.get('RV_BATCH_JOB_DEF'),
        'batch_job_queue': os.environ.get('RV_BATCH_JOB_QUEUE'),
        'github_repo': os.environ.get('RV_GITHUB_REPO')
    }


def _get_from_file(config_uri, profile=None):
    profile = _get_profile(profile=profile)
    try:
        config_str = file_to_str(config_uri)
        config = configparser.ConfigParser()
        config.read_string(config_str)
        if profile in config:
            return config[profile]
        return _get_empty()
    except NotReadableError:
        return _get_empty()


def _get_from_env_file(profile=None):
    profile = _get_profile(profile=profile)
    config_uri = os.environ.get('RV_CONFIG_URI')
    if config_uri is None:
        return _get_empty()
    return _get_from_file(config_uri, profile=profile)


def _merge_dict(base, override):
    new_dict = dict(base)
    for key, val in override.items():
        if val is not None:
            new_dict[key] = val
    return new_dict


def _validate_config(config):
    var_names = ['batch_job_queue', 'batch_job_def', 'github_repo']
    for var_name in var_names:
        if config[var_name] is None:
            raise ValueError('Cannot find {} in RV config'.format(var_name))


def get_rv_config(batch_job_queue=None, batch_job_def=None, github_repo=None,
                  profile=None, home_config_path=HOME_CONFIG_PATH):
    """Get an RV configuration dictionary.

    This computes an RV configuration dictionary from configuration files,
    environment variables, and arguments to this function. It uses a
    precedence hierarchy (similar to the AWS CLI) to determine how values are
    overridden.

    There are two optional configuration files: the environment
    config file (pointed to by env var `RV_CONFIG_URI` which can be remote) and
    the home config file (at ~/.rastervision/config.ini). The files should be
    in the format of a .ini parsable by the built-in Python configparser. They
    should contain sections which are referred to as profiles. The profile that
    is used is determined by the `profile` argument if provided or the
    `RV_PROFILE` env var.

    The optional environment variables include: `RV_BATCH_JOB_DEF`,
    `RV_BATCH_JOB_QUEUE`, and `RV_GITHUB_REPO`.

    The precedence ordering is:
    arguments > env vars > home config file > env config file

    Args:
        batch_job_queue: the AWS Batch job queue name to use when running
            remote jobs
        batch_job_def: the AWS Batch job definition name to use
        github_repo: the Github repo containing the branch of RV to download
            and utilize when running remote jobs
        profile: the section of the config file with fields to use
        home_config_path: path to the home config file

    Returns:
        Dict of form
        {
            'batch_job_queue': '...',
            'batch_job_def': '...',
            'github_repo': '...'
        }
    """
    profile = _get_profile(profile=profile)
    config = _get_from_env_file(profile=profile)
    config = _merge_dict(config,
                         _get_from_file(home_config_path, profile=profile))
    config = _merge_dict(config, _get_from_env())
    config = _merge_dict(config, _get_from_args(
        batch_job_queue=batch_job_queue, batch_job_def=batch_job_def,
        github_repo=github_repo))
    _validate_config(config)
    return config
