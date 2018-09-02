import os
from copy import deepcopy

import rastervision as rv
from rastervision.core import CommandIODefinition
from rastervision.core.config import (Config, ConfigBuilder)
from rastervision.protos.experiment2_pb2 import ExperimentConfig2 as ExperimentConfigMsg

class ExperimentConfig(Config):
    def __init__(self,
                 id,
                 task,
                 backend,
                 dataset,
                 evaluators,
                 analyze_uri,
                 chip_uri,
                 train_uri,
                 predict_uri,
                 eval_uri,
                 bundle_uri,
                 analyzers=None):
        if analyzers is None:
            analyzers = []

        self.id = id
        self.task = task
        self.backend = backend
        self.dataset = dataset
        self.analyzers = analyzers
        self.evaluators = evaluators
        self.analyze_uri = analyze_uri
        self.chip_uri = chip_uri
        self.train_uri = train_uri
        self.predict_uri = predict_uri
        self.eval_uri = eval_uri
        self.bundle_uri = bundle_uri

    def preprocess_command(self, command_type, experiment_config, context=None):
        """
        Returns a tuple (config, dependencies) with the
        """
        io_def = CommandIODefinition()
        new_task, sub_io_def = self.task.preprocess_command(command_type,
                                                            experiment_config,
                                                            context)
        io_def.merge(sub_io_def)

        new_backend, sub_io_def = self.backend.preprocess_command(command_type,
                                                                  experiment_config,
                                                                  context)
        io_def.merge(sub_io_def)

        new_dataset, sub_io_def = self.dataset.preprocess_command(command_type,
                                                                  experiment_config,
                                                                  context)
        io_def.merge(sub_io_def)

        new_analyzers = []
        for analyzer in self.analyzers:
            new_analyzer, sub_io_def = analyzer.preprocess_command(command_type,
                                                                   experiment_config,
                                                                   context)
            io_def.merge(sub_io_def)
            new_analyzers.append(new_analyzer)

        new_evaluators = []
        for evaluator in self.evaluators:
            new_evaluator, sub_io_def = evaluator.preprocess_command(command_type,
                                                                     experiment_config,
                                                                     context)
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

    def to_proto(self):
        analyzers = list(map(lambda a: a.to_proto(),
                             self.analyzers))
        evaluators = list(map(lambda e: e.to_proto(),
                              self.evaluators))

        msg = ExperimentConfigMsg(id=self.id,
                                  task=self.task.to_proto(),
                                  backend=self.backend.to_proto(),
                                  dataset=self.dataset.to_proto(),
                                  analyzers=analyzers,
                                  evaluators=evaluators)
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
    def __init__(self, prev = None):
        config = {}
        if prev:
            config = { "id": prev.id,
                       "task": prev.task,
                       "backend": prev.backend,
                       "dataset": prev.dataset,
                       "analyzers" : prev.analyzers,
                       "evaluators" : prev.evaluators,
                       "analyze_uri": prev.analyze_uri,
                       "chip_uri": prev.chip_uri,
                       "train_uri": prev.train_uri,
                       "predict_uri": prev.predict_uri,
                       "eval_uri": prev.eval_uri,
                       "bundle_uri": prev.bundle_uri}
        super().__init__(ExperimentConfig, config)
        self.root_uri = None # TODO: Store with experiment config?
        self.analyze_key = "default"
        self.chip_key = "default"
        self.train_key = "default"
        self.predict_key = "default"
        self.eval_key = "default"
        self.bundle_key = "default"

    def build(self):
        self.validate()
        # Build any missing paths through
        b = self
        if not self.config.get('analyze_uri'):
            if not self.root_uri:
                raise rv.ConfigError("Need to set root_uri if command uri's not explicitly set.")
            uri = os.path.join(self.root_uri, rv.ANALYZE.lower(), self.analyze_key)
            b = b.with_analyze_uri(uri)
        if not self.config.get('chip_uri'):
            if not self.root_uri:
                raise rv.ConfigError("Need to set root_uri if command uri's not explicitly set.")
            uri = os.path.join(self.root_uri, rv.CHIP.lower(), self.chip_key)
            b = b.with_chip_uri(uri)
        if not self.config.get('train_uri'):
            if not self.root_uri:
                raise rv.ConfigError("Need to set root_uri if command uri's not explicitly set.")
            uri = os.path.join(self.root_uri, rv.TRAIN.lower(), self.train_key)
            b = b.with_train_uri(uri)
        if not self.config.get('predict_uri'):
            if not self.root_uri:
                raise rv.ConfigError("Need to set root_uri if command uri's not explicitly set.")
            uri = os.path.join(self.root_uri, rv.PREDICT.lower(), self.predict_key)
            b = b.with_predict_uri(uri)
        if not self.config.get('eval_uri'):
            if not self.root_uri:
                raise rv.ConfigError("Need to set root_uri if command uri's not explicitly set.")
            uri = os.path.join(self.root_uri, rv.EVAL.lower(), self.eval_key)
            b = b.with_eval_uri(uri)
        if not self.config.get('bundle_uri'):
            if not self.root_uri:
                raise rv.ConfigError("Need to set root_uri if command uri's not explicitly set.")
            uri = os.path.join(self.root_uri, rv.BUNDLE.lower(), self.bundle_key)
            b = b.with_bundle_uri(uri)

        if not self.config.get('task'):
            raise rv.ConfigError("Task needs to be set. Use 'with_task'.")

        if not self.config.get('backend'):
            raise rv.ConfigError("Backend needs to be set. Use 'with_backend'.")

        if not self.config.get('dataset'):
            raise rv.ConfigError("Dataset needs to be set. Use 'with_dataset'.")

        evaluators = self.config.get('evaluators')
        if not evaluators:
            e = rv._registry.get_default_evaluator_provider(self.config['task'].task_type) \
                            .construct(self.config['task'])
            b = b.with_evaluator(e)

        return ExperimentConfig(**b.config)

    def from_proto(self, msg):
        b = ExperimentConfigBuilder()
        analyzers = list(map(lambda a: rv.AnalyzerConfig.from_proto(a),
                             msg.analyzers))
        evaluators = list(map(lambda e: rv.EvaluatorConfig.from_proto(e),
                              msg.evaluators))
        return b.with_id(msg.id) \
                .with_task(rv.TaskConfig.from_proto(msg.task)) \
                .with_backend(rv.BackendConfig.from_proto(msg.backend)) \
                .with_dataset(rv.DatasetConfig.from_proto(msg.dataset)) \
                .with_analyzers(analyzers) \
                .with_evaluators(evaluators) \
                .with_analyze_uri(msg.analyze_uri) \
                .with_chip_uri(msg.chip_uri) \
                .with_train_uri(msg.train_uri) \
                .with_predict_uri(msg.predict_uri) \
                .with_eval_uri(msg.eval_uri) \
                .with_bundle_uri(msg.bundle_uri)

    def with_id(self, id):
        b = deepcopy(self)
        b.config['id'] = id
        return b

    def with_task(self, task):
        b = deepcopy(self)
        b.config['task'] = task
        return b

    def with_backend(self, backend):
        b = deepcopy(self)
        b.config['backend'] = backend
        return b

    def with_dataset(self, dataset):
        b = deepcopy(self)
        b.config['dataset'] = dataset
        return b

    def with_analyzers(self, analyzers):
        b = deepcopy(self)
        b.config['analyzers'] = analyzers
        return b

    def with_analyzer(self, analyzer):
        return self.with_analyzers([analyzer])

    def with_stats_analyzer(self):
        a = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER).build()
        return self.with_analyzer(a)

    def with_evaluators(self, evaluators):
        b = deepcopy(self)
        b.config['evaluators'] = evaluators
        return b

    def with_evaluator(self, evaluator):
        return self.with_evaluators([evaluator])

    def with_analyze_uri(self, uri):
        b = deepcopy(self)
        b.config['analyze_uri'] = uri
        return b

    def with_chip_uri(self, uri):
        b = deepcopy(self)
        b.config['chip_uri'] = uri
        return b

    def with_train_uri(self, uri):
        b = deepcopy(self)
        b.config['train_uri'] = uri
        return b

    def with_predict_uri(self, uri):
        b = deepcopy(self)
        b.config['predict_uri'] = uri
        return b

    def with_eval_uri(self, uri):
        b = deepcopy(self)
        b.config['eval_uri'] = uri
        return b

    def with_bundle_uri(self, uri):
        b = deepcopy(self)
        b.config['bundle_uri'] = uri
        return b

    def with_root_uri(self, uri):
        b = deepcopy(self)
        b.root_uri = uri
        return b

    def with_analyze_key(self, key):
        b = deepcopy(self)
        b.analyze_key = key
        return b

    def with_chip_key(self, key):
        b = deepcopy(self)
        b.chip_key = key
        return b

    def with_train_key(self, key):
        b = deepcopy(self)
        b.train_key = key
        return b

    def with_predict_key(self, key):
        b = deepcopy(self)
        b.predict_key = key
        return b

    def with_eval_key(self, key):
        b = deepcopy(self)
        b.eval_key = key
        return b

    def with_bundle_key(self, key):
        b = deepcopy(self)
        b.bundle_key = key
        return b
