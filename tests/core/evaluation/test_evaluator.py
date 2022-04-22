import unittest

from rastervision.core.evaluation import (EvaluatorConfig)


class MockRVPipelineConfig:
    eval_uri = '/abc/def/eval'


class TestEvaluatorConfig(unittest.TestCase):
    def test_update(self):
        cfg = EvaluatorConfig(output_uri=None)
        pipeline_cfg = MockRVPipelineConfig()
        cfg.update(pipeline_cfg)
        self.assertEqual(cfg.output_uri, pipeline_cfg.eval_uri)
        self.assertEqual(cfg.get_output_uri(),
                         f'{pipeline_cfg.eval_uri}/eval.json')
        self.assertEqual(
            cfg.get_output_uri('group1'),
            f'{pipeline_cfg.eval_uri}/group1/eval.json')

    def test_get_output_uri(self):
        cfg = EvaluatorConfig(output_uri='/abc/def')
        self.assertEqual(cfg.get_output_uri(), '/abc/def/eval.json')
        self.assertEqual(
            cfg.get_output_uri('group1'), '/abc/def/group1/eval.json')


if __name__ == '__main__':
    unittest.main()
