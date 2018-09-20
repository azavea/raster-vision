import unittest

import numpy as np

from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.data.label.object_detection_labels import (
    ObjectDetectionLabels)


class ObjectDetectionLabelsTest(unittest.TestCase):
    def setUp(self):
        self.class_map = ClassMap([ClassItem(1, 'car'), ClassItem(2, 'house')])

        self.npboxes = np.array([
            [0., 0., 2., 2.],
            [2., 2., 4., 4.],
        ])
        self.class_ids = np.array([1, 2])
        self.scores = np.array([0.9, 0.9])
        self.labels = ObjectDetectionLabels(
            self.npboxes, self.class_ids, scores=self.scores)

    def test_from_boxlist(self):
        from object_detection.utils.np_box_list import BoxList

        boxlist = BoxList(self.npboxes)
        boxlist.add_field('classes', self.class_ids)
        boxlist.add_field('scores', self.scores)
        labels = ObjectDetectionLabels.from_boxlist(boxlist)
        labels.assert_equal(self.labels)

    def test_make_empty(self):
        npboxes = np.empty((0, 4))
        class_ids = np.empty((0, ))
        scores = np.empty((0, ))
        expected_labels = ObjectDetectionLabels(
            npboxes, class_ids, scores=scores)

        labels = ObjectDetectionLabels.make_empty()
        labels.assert_equal(expected_labels)

    def test_constructor(self):
        labels = ObjectDetectionLabels(
            self.npboxes, self.class_ids, scores=self.scores)
        expected_labels = ObjectDetectionLabels(self.npboxes, self.class_ids,
                                                self.scores)
        labels.assert_equal(expected_labels)

        labels = ObjectDetectionLabels(self.npboxes, self.class_ids)
        scores = np.ones(self.class_ids.shape)
        expected_labels = ObjectDetectionLabels(
            self.npboxes, self.class_ids, scores=scores)
        labels.assert_equal(expected_labels)

    def test_get_boxes(self):
        boxes = self.labels.get_boxes()
        self.assertEqual(len(boxes), 2)
        np.testing.assert_array_equal(boxes[0].npbox_format(),
                                      self.npboxes[0, :])
        np.testing.assert_array_equal(boxes[1].npbox_format(),
                                      self.npboxes[1, :])

    def test_len(self):
        nb_labels = len(self.labels)
        self.assertEqual(self.npboxes.shape[0], nb_labels)

    def test_local_to_global(self):
        local_npboxes = np.array([[0., 0., 2., 2.], [2., 2., 4., 4.]])
        window = Box.make_square(10, 10, 10)
        global_npboxes = ObjectDetectionLabels.local_to_global(
            local_npboxes, window)

        expected_global_npboxes = np.array([[10., 10., 12., 12.],
                                            [12., 12., 14., 14.]])
        np.testing.assert_array_equal(global_npboxes, expected_global_npboxes)

    def test_global_to_local(self):
        global_npboxes = np.array([[10., 10., 12., 12.], [12., 12., 14., 14.]])
        window = Box.make_square(10, 10, 10)
        local_npboxes = ObjectDetectionLabels.global_to_local(
            global_npboxes, window)

        expected_local_npboxes = np.array([[0., 0., 2., 2.], [2., 2., 4., 4.]])
        np.testing.assert_array_equal(local_npboxes, expected_local_npboxes)

    def test_local_to_normalized(self):
        local_npboxes = np.array([[0., 0., 2., 2.], [2., 2., 4., 4.]])
        window = Box(0, 0, 10, 100)
        norm_npboxes = ObjectDetectionLabels.local_to_normalized(
            local_npboxes, window)

        expected_norm_npboxes = np.array([[0., 0., 0.2, 0.02],
                                          [0.2, 0.02, 0.4, 0.04]])
        np.testing.assert_array_equal(norm_npboxes, expected_norm_npboxes)

    def test_normalized_to_local(self):
        norm_npboxes = np.array([[0., 0., 0.2, 0.02], [0.2, 0.02, 0.4, 0.04]])
        window = Box(0, 0, 10, 100)
        local_npboxes = ObjectDetectionLabels.normalized_to_local(
            norm_npboxes, window)

        expected_local_npboxes = np.array([[0., 0., 2., 2.], [2., 2., 4., 4.]])
        np.testing.assert_array_equal(local_npboxes, expected_local_npboxes)

    def test_get_overlapping(self):
        window = Box.make_square(0, 0, 2.01)
        labels = ObjectDetectionLabels.get_overlapping(self.labels, window)
        labels.assert_equal(self.labels)

        window = Box.make_square(0, 0, 3)
        labels = ObjectDetectionLabels.get_overlapping(
            self.labels, window, ioa_thresh=0.5)
        npboxes = np.array([[0., 0., 2., 2.]])
        class_ids = np.array([1])
        scores = np.array([0.9])
        expected_labels = ObjectDetectionLabels(
            npboxes, class_ids, scores=scores)
        labels.assert_equal(expected_labels)

        window = Box.make_square(0, 0, 3)
        labels = ObjectDetectionLabels.get_overlapping(
            self.labels, window, ioa_thresh=0.1, clip=True)
        expected_npboxes = np.array([
            [0., 0., 2., 2.],
            [2., 2., 3., 3.],
        ])
        expected_labels = ObjectDetectionLabels(
            expected_npboxes, self.class_ids, scores=self.scores)
        labels.assert_equal(expected_labels)

    def test_concatenate(self):
        npboxes = np.array([[4., 4., 5., 5.]])
        class_ids = np.array([2])
        scores = np.array([0.3])
        labels = ObjectDetectionLabels(npboxes, class_ids, scores=scores)
        new_labels = ObjectDetectionLabels.concatenate(self.labels, labels)

        npboxes = np.array([[0., 0., 2., 2.], [2., 2., 4., 4.],
                            [4., 4., 5., 5.]])
        class_ids = np.array([1, 2, 2])
        scores = np.array([0.9, 0.9, 0.3])
        expected_labels = ObjectDetectionLabels(
            npboxes, class_ids, scores=scores)
        new_labels.assert_equal(expected_labels)

    def test_prune_duplicates(self):
        # This first box has a score below score_thresh so it should get
        # pruned. The third box overlaps with the second, but has higher score,
        # so the second one should get pruned. The fourth box overlaps with
        # the second less than merge_thresh, so it should not get pruned.
        npboxes = np.array([[0., 0., 2., 2.], [2., 2., 4., 4.],
                            [2.1, 2.1, 4.1, 4.1], [3.5, 3.5, 5.5, 5.5]])
        class_ids = np.array([1, 2, 1, 2])
        scores = np.array([0.2, 0.9, 0.9, 1.0])
        labels = ObjectDetectionLabels(npboxes, class_ids, scores=scores)
        score_thresh = 0.5
        merge_thresh = 0.5
        pruned_labels = ObjectDetectionLabels.prune_duplicates(
            labels, score_thresh, merge_thresh)

        self.assertEqual(len(pruned_labels), 2)

        expected_npboxes = np.array([[2.1, 2.1, 4.1, 4.1],
                                     [3.5, 3.5, 5.5, 5.5]])
        expected_class_ids = np.array([1, 2])
        expected_scores = np.array([0.9, 1.0])

        # prune_duplicates does not maintain ordering of boxes, so find match
        # between pruned boxes and expected_npboxes.
        pruned_npboxes = pruned_labels.get_npboxes()
        pruned_inds = [None, None]
        for box_ind, box in enumerate(expected_npboxes):
            for pruned_box_ind, pruned_box in enumerate(pruned_npboxes):
                if np.array_equal(pruned_box, box):
                    pruned_inds[box_ind] = pruned_box_ind
        self.assertTrue(np.all(pruned_inds is not None))

        expected_labels = ObjectDetectionLabels(
            expected_npboxes[pruned_inds],
            expected_class_ids[pruned_inds],
            scores=expected_scores[pruned_inds])
        pruned_labels.assert_equal(expected_labels)


if __name__ == '__main__':
    unittest.main()
