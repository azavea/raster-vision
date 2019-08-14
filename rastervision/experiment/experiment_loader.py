import inspect
import os
from importlib import import_module
from fnmatch import fnmatchcase

import rastervision as rv


class LoaderError(Exception):
    pass


class ExperimentLoader:
    def __init__(self,
                 experiment_args=None,
                 experiment_method_prefix='exp',
                 experiment_method_patterns=None,
                 experiment_name_patterns=None):
        if experiment_args is None:
            experiment_args = {}
        self.exp_args = experiment_args
        self.exp_method_prefix = experiment_method_prefix
        self.exp_method_patterns = experiment_method_patterns
        self.exp_name_patterns = experiment_name_patterns

        self._top_level_dir = os.path.abspath(os.curdir)

    def _get_name_from_path(self, path):
        """Gets an importable  name from a path.

        Note: This code is from the python unittest library
        """
        if path == self._top_level_dir:
            return '.'
        path = os.path.splitext(os.path.normpath(path))[0]

        _relpath = os.path.relpath(path, self._top_level_dir)
        assert not os.path.isabs(_relpath), 'Path must be within the project'
        assert not _relpath.startswith('..'), 'Path must be within the project'

        name = _relpath.replace(os.path.sep, '.')
        return name

    def load_from_file(self, path):
        """Loads experiments and commands from an ExperimentSet contained
        in the given file.

        Returns a tuple (experiments, commands)"""
        name = self._get_name_from_path(path)
        return self.load_from_module(name)

    def load_from_module(self, name):
        """Loads experiments and commands from an ExperimentSet contained
        in the given module.

        Returns a tuple (experiments, commands)"""
        experiments, commands = [], []
        module = import_module(name)
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, rv.ExperimentSet):
                experiment_set = obj()
                es, cs = self.load_from_set(experiment_set)
                experiments += es
                commands += cs
        return (experiments, commands)

    def load_from_set(self, experiment_set):
        return self.load_from_sets([experiment_set])

    def load_from_sets(self, experiment_sets):
        experiments = []
        commands = []
        for experiment_set in experiment_sets:
            for attrname in dir(experiment_set):
                include_method = True
                if not attrname.startswith(self.exp_method_prefix):
                    include_method = False
                exp_func = getattr(experiment_set, attrname)
                if not callable(exp_func):
                    include_method = False

                if include_method:
                    full_name = '%s.%s' % (experiment_set.__module__,
                                           exp_func.__qualname__)

                    if self.exp_method_patterns:
                        include_method = any(
                            fnmatchcase(full_name, pattern)
                            for pattern in self.exp_method_patterns)

                    if include_method:
                        es, cs = self.load_from_experiment(exp_func, full_name)
                        experiments.extend(es)
                        commands.extend(cs)

        if self.exp_name_patterns:
            # Avoid running commands if experiment names are used to filter
            commands = []

            def include(e):
                return any(
                    fnmatchcase(e.id, pattern)
                    for pattern in self.exp_name_patterns)

            experiments = list(filter(include, experiments))

        return (experiments, commands)

    def load_from_experiment(self, exp_func, full_name):
        experiments, commands = [], []
        kwargs = {}
        params = inspect.signature(exp_func).parameters.items()
        required_params = [
            key for key, param in params
            if param.default == inspect.Parameter.empty
        ]
        missing_params = set(required_params) - set(self.exp_args.keys())
        if missing_params:
            raise LoaderError('Missing required parameters '
                              'for experiment method '
                              '{}: "{}"'.format(full_name,
                                                '", "'.join(missing_params)))
        for key, param in params:
            if key in self.exp_args:
                kwargs[key] = self.exp_args[key]

        exp = exp_func(**kwargs)
        if not isinstance(exp, list):
            exp = [exp]

        for o in exp:
            if isinstance(o, rv.CommandConfig):
                commands.append(o)
            elif isinstance(o, rv.ExperimentConfig):
                experiments.append(o)
            else:
                raise LoaderError(
                    'Unknown type for experiment or command: {}'.format(
                        type(o)))

        return (experiments, commands)
