import os
import tempfile
from pathlib import Path
import logging
import json

from everett.manager import (ConfigManager, ConfigDictEnv, ConfigEnvFileEnv,
                             ConfigIniEnv, ConfigOSEnv)

from rastervision2.pipeline.verbosity import Verbosity

log = logging.getLogger(__name__)


def load_conf_list(s):
    """Loads a list of items from the config.

    Lists should be comma separated.

    This takes into account that previous versions of Raster Vision
    allowed for a `[ "module" ]` like syntax, even though that didn't
    work for multi-value lists.
    """
    try:
        # A comma separated list of values will be transformed to
        # having a list-like string, with ' instead of ". Replacing
        # single quotes with double quotes lets us parse it as a JSON list.
        return json.loads(s.replace("'", '"'))
    except json.JSONDecodeError:
        return list(map(lambda x: x.strip(), s.split(',')))


class RVConfig:
    DEFAULT_PROFILE = 'default'

    tmp_dir = None

    def __init__(self):
        self.reset()

    def reset(self,
              profile=None,
              rv_home=None,
              config_overrides=None,
              tmp_dir=None,
              verbosity=Verbosity.NORMAL):
        self.verbosity = verbosity

        root_log = logging.getLogger('rastervision2')
        if self.verbosity >= Verbosity.VERBOSE:
            root_log.setLevel(logging.DEBUG)
        elif self.verbosity >= Verbosity.NORMAL:
            root_log.setLevel(logging.INFO)
        else:
            root_log.setLevel(logging.WARN)

        if tmp_dir is not None:
            self.set_tmp_dir(tmp_dir)

        if profile is None:
            if os.environ.get('RV_PROFILE'):
                profile = os.environ.get('RV_PROFILE')
            else:
                profile = RVConfig.DEFAULT_PROFILE

        if config_overrides is None:
            config_overrides = {}

        if rv_home is None:
            home = os.path.expanduser('~')
            rv_home = os.path.join(home, '.rastervision')
        self.rv_home = rv_home

        config_file_locations = self._discover_config_file_locations(profile)

        self.config = ConfigManager(
            [
                ConfigDictEnv(config_overrides),
                ConfigOSEnv(),
                ConfigIniEnv(config_file_locations),
            ],
            doc='Check https://docs.rastervision.io/ for docs.')

    def _discover_config_file_locations(self, profile):
        result = []

        # Allow for user to specify specific config file
        # in the RV_CONFIG env variable.
        env_specified_path = os.environ.get('RV_CONFIG')
        if env_specified_path:
            result.append(env_specified_path)

        # Allow user to specify config directory that will
        # contain profile configs in RV_CONFIG_DIR
        # env variable. Otherwise, use "$HOME/.rastervision"
        env_specified_dir_path = os.environ.get('RV_CONFIG_DIR')
        if env_specified_dir_path:
            result.append(os.path.join(env_specified_dir_path, profile))
        else:
            result.append(os.path.join(self.rv_home, profile))
        result.append(os.path.join(os.getcwd(), '.rastervision'))

        # Filter out any that do not exist.
        results_that_exist = list(filter(lambda x: os.path.exists(x), result))

        # If the profile is not default, and there is no config that exists,
        # then throw an error.
        if not any(results_that_exist) and profile != RVConfig.DEFAULT_PROFILE:
            raise Exception('Configuration Profile {} not found. '
                            'Checked: {}'.format(profile, ', '.join(result)))

        return results_that_exist

    @staticmethod
    def get_tmp_dir():
        if RVConfig.tmp_dir is None:
            RVConfig.set_tmp_dir()
        return tempfile.TemporaryDirectory(dir=RVConfig.tmp_dir)

    @staticmethod
    def get_tmp_dir_root():
        if RVConfig.tmp_dir is None:
            RVConfig.set_tmp_dir()
        return RVConfig.tmp_dir

    @staticmethod
    def set_tmp_dir(tmp_dir=None):
        """Set RVConfig.tmp_dir to well-known value.

        This static method sets the value of RVConfig.tmp_dir to some
        well-known value. The value is chosen from one of the
        following (in order of preference): an explicit value
        (presumably from the command line) is considered first, then
        values from the environment are considered, then the current
        value of RVConfig.tmp_dir is considered, then a directory from
        tempfile.TemporaryDirectory() is considered.

        Args:
            tmp_dir: Either a string or None.
        """
        DEFAULT_DIR = '/opt/data/tmp/'

        # Check the various possibilities in order of priority.

        env_arr = [
            os.environ.get(k) for k in ['TMPDIR', 'TEMP', 'TMP']
            if k in os.environ
        ]

        dir_arr = [tmp_dir] + env_arr + [RVConfig.tmp_dir]
        dir_arr = [d for d in dir_arr if d is not None]
        tmp_dir = dir_arr[0] if dir_arr else tempfile.TemporaryDirectory().name

        try:
            # Try to create directory
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir, exist_ok=True)
            # Check that it is actually a directory
            if not os.path.isdir(tmp_dir):
                raise Exception(
                    '{} is not a directory.'.format(tmp_dir))
            # Can we interact with directory?
            Path.touch(Path(os.path.join(tmp_dir, '.can_touch')))
            # All checks have passed by this point
            RVConfig.tmp_dir = tmp_dir

        # If directory cannot be made and/or cannot be interacted
        # with, fall back to default.
        except Exception as e:
            log.warning(
                'Root temporary directory cannot be used: {}. Using root: {}'.
                format(tmp_dir, DEFAULT_DIR))
            RVConfig.tmp_dir = DEFAULT_DIR
        finally:
            os.makedirs(RVConfig.tmp_dir, exist_ok=True)
            log.debug('Temporary directory is: {}'.format(RVConfig.tmp_dir))

    def get_subconfig(self, namespace):
        return self.config.with_namespace(namespace)

    def get_verbosity(self):
        return self.verbosity

    def get_config_dict(self, rv_config_schema):
        config_dict = {}
        for namespace, keys in rv_config_schema.items():
            for key in keys:
                config_dict[namespace + '_' + key] = \
                    self.get_subconfig(namespace)(key)
        return config_dict
