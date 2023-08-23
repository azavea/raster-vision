import unittest
from os.path import join

from rastervision.pipeline.file_system.utils import get_tmp_dir, file_exists
from rastervision.core.box import Box
from rastervision.core.data import (ClassConfig, IdentityCRSTransformer,
                                    ChipClassificationGeoJSONStore)
from rastervision.core.data.label.chip_classification_labels import (
    ClassificationLabel, ChipClassificationLabels)

from tests import data_file_path


class TestChipClassificationLabels(unittest.TestCase):
    def setUp(self):
        self.labels = ChipClassificationLabels()

        self.cell1 = Box.make_square(0, 0, 2)
        self.class_id1 = 1
        self.labels.set_cell(self.cell1, self.class_id1)

        self.cell2 = Box.make_square(0, 2, 2)
        self.class_id2 = 2
        self.labels.set_cell(self.cell2, self.class_id2)

    def test_get_cell(self):
        cell = Box.make_square(0, 2, 3)
        class_id = self.labels.get_cell_class_id(cell)
        self.assertEqual(class_id, None)

        class_id = self.labels.get_cell_class_id(self.cell1)
        self.assertEqual(class_id, self.class_id1)

        class_id = self.labels.get_cell_class_id(self.cell2)
        self.assertEqual(class_id, self.class_id2)

    def test_get_singleton_labels(self):
        labels = self.labels.get_singleton_labels(self.cell1)

        cells = labels.get_cells()
        self.assertEqual(len(cells), 1)

        class_id = labels.get_cell_class_id(self.cell1)
        self.assertEqual(class_id, self.class_id1)

    def test_get_cells(self):
        cells = self.labels.get_cells()
        self.assertEqual(len(cells), 2)
        # ordering of cells isn't known
        self.assertTrue((cells[0] == self.cell1 and cells[1] == self.cell2)
                        or (cells[1] == self.cell1 and cells[0] == self.cell2))

    def test_get_class_ids(self):
        cells = self.labels.get_cells()
        class_ids = self.labels.get_class_ids()
        # check that order of class_ids corresponds to order of cells
        if (cells[0] == self.cell1 and cells[1] == self.cell2):
            self.assertListEqual(class_ids, [1, 2])
        elif (cells[1] == self.cell1 and cells[0] == self.cell2):
            self.assertListEqual(class_ids, [2, 1])

    def test_extend(self):
        labels = ChipClassificationLabels()
        cell3 = Box.make_square(0, 4, 2)
        class_id3 = 1
        labels.set_cell(cell3, class_id3)

        self.labels.extend(labels)
        cells = self.labels.get_cells()
        self.assertEqual(len(cells), 3)
        self.assertTrue(cell3 in cells)

    def test_filter_by_aoi(self):
        aois = [Box.make_square(0, 0, 2).to_shapely()]
        filt_labels = self.labels.filter_by_aoi(aois)

        exp_labels = ChipClassificationLabels()
        cell1 = Box.make_square(0, 0, 2)
        class_id1 = 1
        exp_labels.set_cell(cell1, class_id1)
        self.assertEqual(filt_labels, exp_labels)

        aois = [Box.make_square(4, 4, 2).to_shapely()]
        filt_labels = self.labels.filter_by_aoi(aois)

        exp_labels = ChipClassificationLabels()
        self.assertEqual(filt_labels, exp_labels)

    def test_len(self):
        self.assertEqual(len(self.labels), 2)

    def test_get_values(self):
        values_expected = [
            ClassificationLabel(class_id=self.class_id1),
            ClassificationLabel(class_id=self.class_id2)
        ]
        self.assertListEqual(self.labels.get_values(), values_expected)

    def test_get_scores(self):
        self.assertIsNone(self.labels.get_cell_scores(self.cell1))
        self.assertIsNone(self.labels.get_cell_scores(Box(0, 0, 10, 10)))

    def test_save(self):
        uri = data_file_path('bboxes.geojson')
        class_config = ClassConfig(names=['1', '2'])
        crs_transformer = IdentityCRSTransformer()
        ls = ChipClassificationGeoJSONStore(
            uri,
            class_config=class_config,
            crs_transformer=crs_transformer,
        )
        labels = ls.get_labels()
        with get_tmp_dir() as tmp_dir:
            save_uri = join(tmp_dir, 'labels.geojson')
            labels.save(save_uri, class_config, crs_transformer)
            self.assertTrue(file_exists(save_uri))


if __name__ == '__main__':
    unittest.main()
