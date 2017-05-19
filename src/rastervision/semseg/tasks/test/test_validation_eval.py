import unittest

import numpy as np

from rastervision.semseg.tasks.validation_eval import Scores


class ScoresTestCase(unittest.TestCase):
    def setUp(self):
        self.label_names = ['a', 'b']
        self.confusion_mat = np.array([
            [1, 3],
            [2, 3]])

        self.scores = Scores()
        self.scores.compute_scores(self.label_names, self.confusion_mat)

    def test_avg_accuracy(self):
        avg_accuracy = 4/9
        self.assertEqual(avg_accuracy, self.scores.avg_accuracy)

    def test_f1(self):
        f1 = [0.2857, 0.5455]
        self.assertTrue(np.allclose(f1, self.scores.f1, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
