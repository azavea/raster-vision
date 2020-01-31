class RegistryError(Exception):
    pass


class Registry():
    def __init__(self):
        self.reset()

    def reset(self):
        self.runners = {}
        self.filesystems = []
        self.configs = {}
        self.config_upgraders = {}
        self.rv_config_schema = {}

    def get_runner(self, runner_type):
        runner = self.runners.get(runner_type)
        if runner:
            return runner
        else:
            RegistryError('{} is not a registered runner.'.format(runner_type))

    def get_file_system(self, uri: str, mode: str = 'r'):
        for fs in self.filesystems:
            if fs.matches_uri(uri, mode):
                return fs
        if mode == 'w':
            raise RegistryError('No matching filesystem to handle '
                                'writing to uri {}'.format(uri))
        else:
            raise RegistryError('No matching filesystem to handle '
                                'reading from uri {}'.format(uri))

    def get_config(self, type_hint):
        config = self.configs.get(type_hint)
        if config:
            return config
        else:
            raise RegistryError(
                '{} is not a registered config type hint.'.format(type_hint))

    def get_config_upgraders(self, type_hint):
        out = self.config_upgraders.get(type_hint)
        if out:
            return out
        else:
            raise RegistryError(
                '{} is not a registered config upgrader type hint.'.format(
                    type_hint))

    def add_config(self, type_hint, config_cls, version=0, upgraders=None):
        if type_hint in self.configs:
            raise RegistryError(
                'There is already a config registered for type_hint {}'.format(
                    type_hint))

        self.configs[type_hint] = config_cls

        if type_hint in self.config_upgraders:
            raise RegistryError(
                'There are already config upgraders registered for type_hint {}'.
                format(type_hint))
        self.config_upgraders[type_hint] = (version, upgraders)
