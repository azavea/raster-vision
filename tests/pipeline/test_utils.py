import unittest

from rastervision.pipeline.utils import split_into_groups


class TestUtils(unittest.TestCase):
    def test_split_into_groups(self):
        lst = [1, 2, 3, 4, 5, 6]

        g1 = split_into_groups(lst[:5], 3)
        self.assertEqual(g1, [[1, 2], [3, 4], [5]])

        g2 = split_into_groups(lst, 7)
        self.assertEqual(g2, [[1], [2], [3], [4], [5], [6]])

        g3 = split_into_groups(lst[0:1], 7)
        self.assertEqual(g3, [[1]])

        g4 = split_into_groups(lst, 3)
        self.assertEqual(g4, [[1, 2], [3, 4], [5, 6]])


if __name__ == '__main__':
    unittest.main()
