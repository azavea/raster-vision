from typing import Any, Dict, List, Optional
import os
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
import json

from everett.manager import (ConfigManager, ConfigDictEnv, ConfigOSEnv,
                             ConfigurationMissingError)
from everett.ext.inifile import ConfigIniEnv

from rastervision.pipeline.verbosity import Verbosity

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


# TODO change name to SystemConfig so it's not tied to RV?
class RVConfig:
    """A store of global user-specific configuration not tied to particular pipelines.

    This is used to store user-specific configuration like the root temporary
    directory, verbosity, and other system-wide configuration handled by Everett
    (eg. which AWS Batch job queue to use).

    Attributes:
        DEFAULT_PROFILE: the default RV configuration profile name
        DEFAULT_TMP_DIR_ROOT: the default location for root of temporary directories
    """
    DEFAULT_PROFILE: str = 'default'
    DEFAULT_TMP_DIR_ROOT: str = '/opt/data/tmp'

    def __init__(self):
        self.set_verbosity()
        self.set_tmp_dir_root()
        self.set_everett_config()

    def set_verbosity(self, verbosity: Verbosity = Verbosity.NORMAL):
        """Set verbosity level for logging."""
        self.verbosity = verbosity
        root_log = logging.getLogger('rastervision')
        if self.verbosity >= Verbosity.VERBOSE:
            root_log.setLevel(logging.DEBUG)
        elif self.verbosity >= Verbosity.NORMAL:
            root_log.setLevel(logging.INFO)
        else:
            root_log.setLevel(logging.WARN)

    def get_verbosity(self) -> Verbosity:
        """Returns verbosity level for logging."""
        return self.verbosity

    def get_verbosity_cli_opt(self) -> str:
        """Returns verbosity in a form that can be passed to RV CLI cmds.

        Returns:
            str: string like "-vvv...".
        """
        num_vs = max(0, self.get_verbosity() - 1)
        if num_vs == 0:
            return ''
        return f'-{"v" * num_vs}'

    def get_tmp_dir(self) -> TemporaryDirectory:
        """Return a new TemporaryDirectory object."""
        return TemporaryDirectory(dir=self.tmp_dir_root)

    def get_tmp_dir_root(self) -> str:
        """Return the root of all temp dirs."""
        return self.tmp_dir_root

    def set_tmp_dir_root(self, tmp_dir_root: Optional[str] = None):
        """Set root of all temporary directories.

        To set the value, the following rules are used in decreasing priority:

        1) the ``tmp_dir_root`` argument if it is not ``None``
        2) an environment variable (``TMPDIR``, ``TEMP``, or ``TMP``)
        3) a default temporary directory which is a directory returned by
           :class:`tempfile.TemporaryDirectory`
        """
        # Check the various possibilities in order of priority.
        env_arr = [
            os.environ.get(k) for k in ['TMPDIR', 'TEMP', 'TMP']
            if k in os.environ
        ]

        dir_arr = [tmp_dir_root] + env_arr + [RVConfig.DEFAULT_TMP_DIR_ROOT]
        dir_arr = [d for d in dir_arr if d is not None]
        tmp_dir_root = dir_arr[0]

        try:
            # Try to create directory
            if not os.path.exists(tmp_dir_root):
                os.makedirs(tmp_dir_root, exist_ok=True)
            # Check that it is actually a directory
            if not os.path.isdir(tmp_dir_root):
                raise Exception('{} is not a directory.'.format(tmp_dir_root))
            # Can we interact with directory?
            Path.touch(Path(os.path.join(tmp_dir_root, '.can_touch')))
            # All checks have passed by this point
            self.tmp_dir_root = tmp_dir_root

        # If directory cannot be made and/or cannot be interacted
        # with, fall back to default system location.
        except Exception:
            system_tmp_dir = TemporaryDirectory().name
            log.warning(
                'Root temporary directory cannot be used: {}. Using root: {}'.
                format(tmp_dir_root, system_tmp_dir))
            self.tmp_dir_root = system_tmp_dir
        finally:
            os.makedirs(self.tmp_dir_root, exist_ok=True)
            log.debug('Temporary directory root is: {}'.format(
                self.tmp_dir_root))

    def get_cache_dir(self) -> TemporaryDirectory:
        """Return the cache directory."""
        cache_dir = os.path.join(self.tmp_dir_root, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def set_everett_config(self,
                           profile: str = None,
                           rv_home: str = None,
                           config_overrides: Dict[str, str] = None):
        """Set Everett config.

        This sets up any other configuration using the Everett library.
        See https://everett.readthedocs.io/

        It roughly mimics the behavior of how the AWS CLI is configured, if that
        is a helpful analogy. Configuration can be specified through configuration
        files, environment variables, and the config_overrides argument in increasing
        order of precedence.

        Configuration files are in the following format:
        ```
        [namespace_1]
        key_11=val_11
        ...
        key_1n=val_1n

        ...

        [namespace_m]
        key_m1=val_m1
        ...
        key_mn=val_mn
        ```

        Each namespace can be used for the configuration of a different plugin.
        Each configuration file is a "profile" with the name of the file being the name
        of the profile. This supports switching between different configuration sets.
        The corresponding environment variable setting for namespace_i and key_ij is
        `namespace_i_key_ij=val_ij`.

        Args:
            profile: name of the RV configuration profile to use. If not set, defaults
                to value of RV_PROFILE env var, or DEFAULT_PROFILE.
            rv_home: a local dir with RV configuration files. If not set, attempts to
                use ~/.rastervision.
            config_overrides: any configuration to override. Each key is of form
                namespace_i_key_ij with corresponding value val_ij.
        """
        if profile is None:
            if os.environ.get('RV_PROFILE'):
                profile = os.environ.get('RV_PROFILE')
            else:
                profile = RVConfig.DEFAULT_PROFILE
        self.profile = profile

        if config_overrides is None:
            config_overrides = {}

        if rv_home is None:
            home = os.path.expanduser('~')
            rv_home = os.path.join(home, '.rastervision')
        self.rv_home = rv_home

        config_file_locations = self._discover_config_file_locations(
            self.profile)
        config_ini_env = ConfigIniEnv(config_file_locations)

        self.config = ConfigManager(
            [
                ConfigOSEnv(),
                ConfigDictEnv(config_overrides),
                config_ini_env,
            ],
            doc=(
                'Check https://docs.rastervision.io/ for docs. '
                'Switch to the version being run and search for Raster Vision '
                'Configuration.'))

    def get_namespace_config(self, namespace: str) -> ConfigManager:
        """Get the key-val pairs associated with a namespace."""
        return self.config.with_namespace(namespace)

    def get_namespace_option(self,
                             namespace: str,
                             key: str,
                             default: Optional[Any] = None,
                             as_bool: bool = False) -> Optional[Any]:
        """Get the value of an option from a namespace."""
        namespace_options = self.config.with_namespace(namespace)
        try:
            val: str = namespace_options(key)
            if as_bool:
                val = val.lower() in ('1', 'true', 'y', 'yes')
            return val
        except ConfigurationMissingError:
            if as_bool:
                return bool(default)
            return default

    def get_config_dict(
            self, rv_config_schema: Dict[str, List[str]]) -> Dict[str, str]:
        """Get all Everett configuration.

        This method is used to serialize an Everett configuration so it can be used on
        a remote instance.

        Args:
            rv_config_schema: each key is a namespace; each value is list of keys within
                that namespace

        Returns:
            Each key is of form namespace_i_key_ij with corresponding value val_ij.
        """
        config_dict = {}
        for namespace, keys in rv_config_schema.items():
            for key in keys:
                try:
                    namespace_options = self.get_namespace_config(namespace)
                    full_key = f'{namespace}_{key}'
                    config_dict[full_key] = namespace_options(key)
                except ConfigurationMissingError:
                    pass

        return config_dict

    def _discover_config_file_locations(self, profile) -> List[str]:
        """Discover the location of RV config files.

        Args:
            profile: the name of the RV profile to use

        Returns:
            a list of paths to RV config files matching the profile name
        """
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
