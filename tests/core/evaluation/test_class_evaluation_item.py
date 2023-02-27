import unittest

import numpy as np

from rastervision.core.evaluation import ClassEvaluationItem


class TestClassEvaluationItem(unittest.TestCase):
    def setUp(self):
        pass

    def test_with_tn(self):
        conf_mat = np.random.randint(100, size=(2, 2))
        [[tn, fp], [fn, tp]] = conf_mat
        item = ClassEvaluationItem(
            class_id=0, class_name='abc', tp=tp, fp=fp, fn=fn, tn=tn)
        self.assertEqual(item.true_pos, tp)
        self.assertEqual(item.true_neg, tn)
        self.assertEqual(item.false_pos, fp)
        self.assertEqual(item.false_neg, fn)

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        self.assertEqual(item.gt_count, fn + tp)
        self.assertEqual(item.pred_count, fp + tp)
        self.assertEqual(item.recall, recall)
        self.assertEqual(item.sensitivity, recall)
        self.assertEqual(item.precision, precision)
        self.assertEqual(item.specificity, tn / (tn + fp))
        self.assertEqual(item.f1,
                         2 * (precision * recall) / (precision + recall))

        json = item.to_json()
        self.assertEqual(json['relative_frequency'],
                         (fn + tp) / (fn + tp + fp + tn))
        self.assertEqual(json['count_error'], abs((fn + tp) - (fp + tp)))
        np.testing.assert_array_equal(np.array(json['conf_mat']), conf_mat)

    def test_without_tn(self):
        conf_mat = np.random.randint(100, size=(2, 2))
        [[_, fp], [fn, tp]] = conf_mat
        tn = None
        item = ClassEvaluationItem(
            class_id=0, class_name='abc', tp=tp, fp=fp, fn=fn, tn=tn)
        self.assertEqual(item.true_pos, tp)
        self.assertEqual(item.true_neg, None)
        self.assertEqual(item.false_pos, fp)
        self.assertEqual(item.false_neg, fn)

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        self.assertEqual(item.gt_count, fn + tp)
        self.assertEqual(item.pred_count, fp + tp)
        self.assertEqual(item.recall, recall)
        self.assertEqual(item.sensitivity, recall)
        self.assertEqual(item.precision, precision)
        self.assertEqual(item.specificity, None)
        self.assertEqual(item.f1,
                         2 * (precision * recall) / (precision + recall))

        json = item.to_json()
        self.assertNotIn('relative_frequency', json)
        self.assertIsNone(json['metrics']['specificity'])
        self.assertEqual(json['true_pos'], tp)
        self.assertEqual(json['false_pos'], fp)
        self.assertEqual(json['false_neg'], fn)

    def test_merge(self):
        conf_mat1 = np.random.randint(100, size=(2, 2))
        [[tn, fp], [fn, tp]] = conf_mat1
        item1 = ClassEvaluationItem(
            class_id=0, class_name='abc', tp=tp, fp=fp, fn=fn, tn=tn)

        conf_mat2 = np.random.randint(100, size=(2, 2))
        [[tn, fp], [fn, tp]] = conf_mat2
        item2 = ClassEvaluationItem(
            class_id=0, class_name='abc', tp=tp, fp=fp, fn=fn, tn=tn)

        item1.merge(item2)
        np.testing.assert_array_equal(item1.conf_mat, conf_mat1 + conf_mat2)

        item3 = ClassEvaluationItem(
            class_id=1, class_name='def', tp=tp, fp=fp, fn=fn, tn=tn)
        self.assertRaises(ValueError, lambda: item1.merge(item3))

    def test_extra_info(self):
        conf_mat = np.random.randint(100, size=(2, 2))
        [[tn, fp], [fn, tp]] = conf_mat
        item = ClassEvaluationItem(
            class_id=0,
            class_name='abc',
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            extra1='extra1',
            extra2='extra2')
        json = item.to_json()
        self.assertEqual(json['extra1'], 'extra1')
        self.assertEqual(json['extra2'], 'extra2')

    def from_multiclass_conf_mat(self):
        conf_mat = np.random.randint(100, size=(10, 10))
        item = ClassEvaluationItem.from_multiclass_conf_mat(
            class_id=3,
            class_name='abc',
            conf_mat=conf_mat,
            extra1='extra1',
            extra2='extra2')

        tp = conf_mat[3, 3]
        fp = conf_mat[:, 3].sum() - tp
        fn = conf_mat[3, :].sum() - tp
        tn = conf_mat.sum() - tp - fp - fn
        self.assertEqual(item.true_pos, tp)
        self.assertEqual(item.false_pos, fp)
        self.assertEqual(item.false_neg, fn)
        self.assertEqual(item.true_neg, tn)

        json = item.to_json()
        self.assertEqual(json['extra1'], 'extra1')
        self.assertEqual(json['extra2'], 'extra2')


if __name__ == '__main__':
    unittest.main()
