import os
from copy import deepcopy

import rastervision as rv
from rastervision.core import CommandIODefinition
from rastervision.core.config import (Config, ConfigBuilder)
from rastervision.utils.files import save_json_config
from rastervision.protos.experiment_pb2 \
    import ExperimentConfig as ExperimentConfigMsg


class ExperimentConfig(Config):
    def __init__(self,
                 id,
                 task,
                 backend,
                 dataset,
                 root_uri,
                 analyze_uri,
                 chip_uri,
                 train_uri,
                 predict_uri,
                 eval_uri,
                 bundle_uri,
                 evaluators=None,
                 analyzers=None):
        if analyzers is None:
            analyzers = []

        self.id = id
        self.task = task
        self.backend = backend
        self.dataset = dataset
        self.analyzers = analyzers
        self.evaluators = evaluators
        self.root_uri = root_uri
        self.analyze_uri = analyze_uri
        self.chip_uri = chip_uri
        self.train_uri = train_uri
        self.predict_uri = predict_uri
        self.eval_uri = eval_uri
        self.bundle_uri = bundle_uri

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        """
        Returns a tuple (config, dependencies) with the
        """
        io_def = CommandIODefinition()
        new_task, sub_io_def = self.task.update_for_command(
            command_type, experiment_config, context)
        io_def.merge(sub_io_def)

        new_backend, sub_io_def = self.backend.update_for_command(
            command_type, experiment_config, context)
        io_def.merge(sub_io_def)

        new_dataset, sub_io_def = self.dataset.update_for_command(
            command_type, experiment_config, context)
        io_def.merge(sub_io_def)

        new_analyzers = []
        for analyzer in self.analyzers:
            new_analyzer, sub_io_def = analyzer.update_for_command(
                command_type, experiment_config, context)
            io_def.merge(sub_io_def)
            new_analyzers.append(new_analyzer)

        new_evaluators = []
        for evaluator in self.evaluators:
            new_evaluator, sub_io_def = evaluator.update_for_command(
                command_type, experiment_config, context)
            io_def.merge(sub_io_def)
            new_evaluators.append(new_evaluator)

        new_config = self.to_builder() \
                         .with_task(new_task) \
                         .with_backend(new_backend) \
                         .with_dataset(new_dataset) \
                         .with_analyzers(new_analyzers) \
                         .with_evaluators(new_evaluators) \
                         .build()

        return (new_config, io_def)

    def make_command_config(self, command_type):
        return rv._registry.get_command_config_builder(command_type)() \
                           .with_experiment(self) \
                           .build()

    def fully_resolve(self):
        """Returns a fully resolved version of this  experiment.

        A fully resolved experiment has all implicit paths put into place,
        and is constructed by calling update_for_command for each command.
        """
        e = self
        for command_type in rv.ALL_COMMANDS:
            e, _ = e.update_for_command(command_type, e)
        return e

    def save_config(self):
        msg = self.to_proto()
        uri = os.path.join(self.root_uri, 'experiments',
                           '{}.json'.format(self.id))
        save_json_config(msg, uri)

    def to_proto(self):
        analyzers = list(map(lambda a: a.to_proto(), self.analyzers))
        evaluators = list(map(lambda e: e.to_proto(), self.evaluators))

        msg = ExperimentConfigMsg(
            id=self.id,
            task=self.task.to_proto(),
            backend=self.backend.to_proto(),
            dataset=self.dataset.to_proto(),
            analyzers=analyzers,
            evaluators=evaluators)
        msg.root_uri = self.root_uri
        msg.analyze_uri = self.analyze_uri
        msg.chip_uri = self.chip_uri
        msg.train_uri = self.train_uri
        msg.predict_uri = self.predict_uri
        msg.eval_uri = self.eval_uri
        msg.bundle_uri = self.bundle_uri
        return msg

    def to_builder(self):
        return ExperimentConfigBuilder(self)

    @staticmethod
    def builder():
        return ExperimentConfigBuilder()

    @staticmethod
    def from_proto(msg):
        """Creates an ExperimentConfig from the specificed protobuf message
        """
        return ExperimentConfigBuilder().from_proto(msg).build()


class ExperimentConfigBuilder(ConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'id': prev.id,
                'task': prev.task,
                'backend': prev.backend,
                'dataset': prev.dataset,
                'analyzers': prev.analyzers,
                'evaluators': prev.evaluators,
                'root_uri': prev.root_uri,
                'analyze_uri': prev.analyze_uri,
                'chip_uri': prev.chip_uri,
                'train_uri': prev.train_uri,
                'predict_uri': prev.predict_uri,
                'eval_uri': prev.eval_uri,
                'bundle_uri': prev.bundle_uri
            }
        super().__init__(ExperimentConfig, config)
        self.analyze_key = None
        self.chip_key = None
        self.train_key = None
        self.predict_key = None
        self.eval_key = None
        self.bundle_key = None

    def validate(self):

        if not self.config.get('root_uri'):
            raise rv.ConfigError('root_uri must be set. Use "with_root_uri"')

        for key in ['task', 'backend', 'dataset', 'id']:
            if self.config.get(key) is None:
                raise rv.ConfigError(
                    'Experiment %s must be set. Use "with_%s".' % (key, key))

    def build(self):
        self.validate()
        # Build any missing paths through
        b = self

        if not self.config.get('analyze_uri'):
            if not self.analyze_key:
                self.analyze_key = self.config['id']
            uri = os.path.join(self.config['root_uri'], rv.ANALYZE.lower(),
                               self.analyze_key)
            b = b.with_analyze_uri(uri)
        if not self.config.get('chip_uri'):
            if not self.chip_key:
                self.chip_key = self.config['id']
            uri = os.path.join(self.config['root_uri'], rv.CHIP.lower(),
                               self.chip_key)
            b = b.with_chip_uri(uri)
        if not self.config.get('train_uri'):
            if not self.train_key:
                self.train_key = self.config['id']
            uri = os.path.join(self.config['root_uri'], rv.TRAIN.lower(),
                               self.train_key)
            b = b.with_train_uri(uri)
        if not self.config.get('predict_uri'):
            if not self.predict_key:
                self.predict_key = self.config['id']
            uri = os.path.join(self.config['root_uri'], rv.PREDICT.lower(),
                               self.predict_key)
            b = b.with_predict_uri(uri)
        if not self.config.get('eval_uri'):
            if not self.eval_key:
                self.eval_key = self.config['id']
            uri = os.path.join(self.config['root_uri'], rv.EVAL.lower(),
                               self.eval_key)
            b = b.with_eval_uri(uri)
        if not self.config.get('bundle_uri'):
            if not self.bundle_key:
                self.bundle_key = self.config['id']
            uri = os.path.join(self.config['root_uri'], rv.BUNDLE.lower(),
                               self.bundle_key)
            b = b.with_bundle_uri(uri)

        evaluators = self.config.get('evaluators')
        if not evaluators:
            task_type = self.config['task'].task_type
            e = rv._registry.get_evaluator_default_provider(task_type) \
                            .construct(self.config['task'])
            b = b.with_evaluator(e)

        return ExperimentConfig(**b.config)

    def from_proto(self, msg):
        b = ExperimentConfigBuilder()
        analyzers = list(
            map(lambda a: rv.AnalyzerConfig.from_proto(a), msg.analyzers))
        evaluators = list(
            map(lambda e: rv.EvaluatorConfig.from_proto(e), msg.evaluators))
        return b.with_id(msg.id) \
                .with_task(rv.TaskConfig.from_proto(msg.task)) \
                .with_backend(rv.BackendConfig.from_proto(msg.backend)) \
                .with_dataset(rv.DatasetConfig.from_proto(msg.dataset)) \
                .with_analyzers(analyzers) \
                .with_evaluators(evaluators) \
                .with_root_uri(msg.root_uri) \
                .with_analyze_uri(msg.analyze_uri) \
                .with_chip_uri(msg.chip_uri) \
                .with_train_uri(msg.train_uri) \
                .with_predict_uri(msg.predict_uri) \
                .with_eval_uri(msg.eval_uri) \
                .with_bundle_uri(msg.bundle_uri)

    def with_id(self, id):
        """Sets an id for the experiment."""
        b = deepcopy(self)
        b.config['id'] = id
        return b

    def with_task(self, task):
        """Sets a specific task type.

        Args:
            task:  A TaskConfig object.

        """
        b = deepcopy(self)
        b.config['task'] = task
        return b

    def with_backend(self, backend):
        """Specifies the backend to be used, e.g. rv.TF_DEEPLAB."""
        b = deepcopy(self)
        b.config['backend'] = backend
        return b

    def with_dataset(self, dataset):
        """Specifies the dataset to be used."""
        b = deepcopy(self)
        b.config['dataset'] = dataset
        return b

    def with_analyzers(self, analyzers):
        """Add analyzers to be used in the analysis stage."""
        b = deepcopy(self)
        b.config['analyzers'] = analyzers
        return b

    def with_analyzer(self, analyzer):
        """Add an analyzer to be used in the analysis stage."""
        return self.with_analyzers([analyzer])

    def with_stats_analyzer(self):
        """Add a stats analyzer to be used in the analysis stage."""
        a = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER).build()
        return self.with_analyzer(a)

    def with_evaluators(self, evaluators):
        """Sets the evaluators to use for the evaluation stage."""
        b = deepcopy(self)
        b.config['evaluators'] = evaluators
        return b

    def with_evaluator(self, evaluator):
        """Sets the evaluator to use for the evaluation stage."""
        return self.with_evaluators([evaluator])

    def with_root_uri(self, uri):
        """Sets the root directory where all output will be stored unless
        subsequently overridden.

        """
        b = deepcopy(self)
        b.config['root_uri'] = uri
        return b

    def with_analyze_uri(self, uri):
        """Sets the location where the results of the analysis stage will be
           stored.

        """
        b = deepcopy(self)
        b.config['analyze_uri'] = uri
        return b

    def with_chip_uri(self, uri):
        """Sets the location where the results of the "chip" stage will be
           stored.

        """
        b = deepcopy(self)
        b.config['chip_uri'] = uri
        return b

    def with_train_uri(self, uri):
        """Sets the location where the results of the training stage will be
           stored.

        """
        b = deepcopy(self)
        b.config['train_uri'] = uri
        return b

    def with_predict_uri(self, uri):
        """Sets the location where the results of the prediction stage will be
           stored.

        """
        b = deepcopy(self)
        b.config['predict_uri'] = uri
        return b

    def with_eval_uri(self, uri):
        """Sets the location where the results of the evaluation stage will be
           stored.

        """
        b = deepcopy(self)
        b.config['eval_uri'] = uri
        return b

    def with_bundle_uri(self, uri):
        """Sets the location where the results of the bundling stage will be
           stored.

        """
        b = deepcopy(self)
        b.config['bundle_uri'] = uri
        return b

    def with_analyze_key(self, key):
        """Sets the key associated with the analysis stage."""
        b = deepcopy(self)
        b.analyze_key = key
        return b

    def with_chip_key(self, key):
        """Sets the key associated with the "chip" stage."""
        b = deepcopy(self)
        b.chip_key = key
        return b

    def with_train_key(self, key):
        """Sets the key associated with the training stage."""
        b = deepcopy(self)
        b.train_key = key
        return b

    def with_predict_key(self, key):
        """Sets the key associated with the prediction stage."""
        b = deepcopy(self)
        b.predict_key = key
        return b

    def with_eval_key(self, key):
        """Sets the key associated with the evaluation stage."""
        b = deepcopy(self)
        b.eval_key = key
        return b

    def with_bundle_key(self, key):
        """Sets the key associated with the bundling stage."""
        b = deepcopy(self)
        b.bundle_key = key
        return b
