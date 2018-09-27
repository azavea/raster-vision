import unittest

import rastervision as rv


class TestClassificationEvaluation(unittest.TestCase):
    def test_missing_config_class_map(self):
        with self.assertRaises(rv.ConfigError):
            rv.evaluation.ClassificationEvaluatorConfig.builder(
                rv.CHIP_CLASSIFICATION_EVALUATOR).build()

    def test_no_missing_config(self):
        try:
            rv.evaluation.ClassificationEvaluatorConfig.builder(
                rv.CHIP_CLASSIFICATION_EVALUATOR).with_class_map(['']).build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')


if __name__ == '__main__':
    unittest.main()
