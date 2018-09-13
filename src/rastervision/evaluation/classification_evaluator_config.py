import os
from copy import deepcopy

import rastervision as rv
from rastervision.evaluation \
    import (EvaluatorConfig, EvaluatorConfigBuilder)
from rastervision.protos.evaluator_pb2 import EvaluatorConfig as EvaluatorConfigMsg
from rastervision.task.utils import (construct_class_map,
                                     classes_to_class_items)


class ClassificationEvaluatorConfig(EvaluatorConfig):
    """Abstract class for usage with simple evaluators that
    are classification-based.
    """

    def __init__(self, evaluator_type, class_map, output_uri=None):
        super().__init__(evaluator_type)
        self.class_map = class_map
        self.output_uri = output_uri

    def to_proto(self):
        sub_msg = EvaluatorConfigMsg.ClassificationEvaluatorConfig(
            class_items=classes_to_class_items(self.class_map),
            output_uri=self.output_uri)
        msg = EvaluatorConfigMsg(
            evaluator_type=self.evaluator_type, classification_config=sub_msg)

        return msg

    def preprocess_command(self, command_type, experiment_config, context=[]):
        conf = self
        io_def = rv.core.CommandIODefinition()
        if command_type == rv.EVAL:
            if not self.output_uri:
                output_uri = os.path.join(experiment_config.eval_uri,
                                          "eval.json")
                conf = conf.to_builder() \
                           .with_output_uri(output_uri) \
                           .build()
            io_def.add_output(conf.output_uri)
        return (conf, io_def)


class ClassificationEvaluatorConfigBuilder(EvaluatorConfigBuilder):
    def __init__(self, cls, prev=None):
        config = {}
        if prev:
            config = {
                "output_uri": prev.output_uri,
                "class_map": prev.class_map
            }
        super().__init__(cls, config)

    @classmethod
    def from_proto(cls, msg):
        b = cls()
        class_map = construct_class_map(
            list(msg.classification_config.class_items))
        return b.with_output_uri(msg.classification_config.output_uri) \
                .with_class_map(class_map)

    def with_output_uri(self, output_uri):
        """Set the output_uri.

            Args:
                output_uri: URI to the stats json to use
        """
        b = deepcopy(self)
        b.config['output_uri'] = output_uri
        return b

    def with_task(self, task):
        if not hasattr(task, 'class_map'):
            raise rv.ConfigError("This evaluator requires a task "
                                 "that has a class_map property")
        return self.with_class_map(task.class_map)

    def with_class_map(self, class_map):
        b = deepcopy(self)
        b.config['class_map'] = class_map
        return b
