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
        name = self._get_name_from_path(path)
        return self.load_from_module(name)

    def load_from_module(self, name):
        result = []
        module = import_module(name)
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, rv.ExperimentSet):
                experiment_set = obj()
                result += self.load_from_set(experiment_set)
        return result

    def load_from_set(self, experiment_set):
        return self.load_from_sets([experiment_set])

    def load_from_sets(self, experiment_sets):
        results = []
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
                        es = self.load_from_experiment(exp_func, full_name)
                        results.extend(es)

        if self.exp_name_patterns:

            def include(e):
                return any(
                    fnmatchcase(e.id, pattern)
                    for pattern in self.exp_name_patterns)

            results = list(filter(include, results))
        return results

    def load_from_experiment(self, exp_func, full_name):
        results = []
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
        if isinstance(exp, list):
            results.extend(exp)
        else:
            results.append(exp)
        return results
