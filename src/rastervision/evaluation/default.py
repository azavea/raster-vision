from abc import (ABC, abstractmethod)

import rastervision as rv


# TODO : DefaultProvider instead  of DefaultXProvider
class DefaultEvaluatorProvider(ABC):
    @staticmethod
    @abstractmethod
    def is_default_for(task_type):
        """Returns True if this evaluator is the default for this tasks_type"""
        pass

    @abstractmethod
    def construct(task):
        """Constructs the default evaluator.
        """
        pass


class DefaultObjectDetectioneEvaluatorProvider(DefaultEvaluatorProvider):
    @staticmethod
    def is_default_for(task_type):
        return task_type == rv.OBJECT_DETECTION

    @staticmethod
    def construct(task):
        return rv.EvaluatorConfig.builder(rv.OBJECT_DETECTION_EVALUATOR) \
                                 .with_task(task) \
                                 .build()


class DefaultChipClassificationEvaluatorProvider(DefaultEvaluatorProvider):
    @staticmethod
    def is_default_for(task_type):
        return task_type == rv.CHIP_CLASSIFICATION

    @staticmethod
    def construct(task):
        return rv.EvaluatorConfig.builder(rv.CHIP_CLASSIFICATION_EVALUATOR) \
                                 .with_task(task) \
                                 .build()
