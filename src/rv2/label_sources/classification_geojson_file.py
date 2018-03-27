import json

import numpy as np

from rv2.labels.classification_labels import (
    ClassificationLabels)
from rv2.labels.object_detection_labels import ObjectDetectionLabels
from rv2.utils.files import file_to_str
from rv2.label_sources.classification_label_source import (
        ClassificationLabelSource)


def load_geojson(geojson, crs_transformer, extent, options):
    """Construct ClassificationLabels from GeoJSON.

    Args:
        options: ClassificationGeoJSONFile.Options
    """
    labels = ClassificationLabels()
    # Use the ObjectDetectionLabels to parse bounding boxes out of the GeoJSON
    # which contains polygons and infer which boxes lie within cells.
    od_labels = ObjectDetectionLabels.from_geojson(
        geojson, crs_transformer)

    cells = extent.get_windows(options.cell_size, options.cell_size)
    for cell in cells:
        # Figure out which class_id to assocate with a cell.
        cell_od_labels = \
            od_labels.get_subwindow(cell, ioa_thresh=options.ioa_thresh)
        cell_boxes = cell_od_labels.get_boxes()
        cell_class_ids = cell_od_labels.get_class_ids()

        # If there are no boxes in the window, we use the
        # background_class_id which can be set to None.
        class_id = (None if options.background_class_id == 0
                    else options.background_class_id)

        if len(cell_class_ids) > 0:
            # When there are boxes associated with more than one class
            # inside the window, we need a way to pick *the* class for
            # that window.
            if options.pick_min_class_id:
                class_id = min(cell_class_ids)
            else:
                biggest_box_ind = np.argmax(
                    [box.get_area() for box in cell_boxes])
                class_id = cell_class_ids[biggest_box_ind]

        labels.set_cell(cell, class_id)

    return labels


class ClassificationGeoJSONFile(ClassificationLabelSource):
    """A GeoJSON file with classification labels in it.

    Ideally the GeoJSON file contains a square for each cell in the grid. But
    in reality, it can be difficult to label imagery in such an exhaustive way.
    So, this can also handle GeoJSON files with non-overlapping polygons that
    do not necessarily cover the entire extent. It infers the grid of cells
    and associated class_ids using the extent and options.

    Args:
        options: ClassificationGeoJSONFile.Options
    """
    def __init__(self, uri, crs_transformer, extent, options, writable=False):
        self.uri = uri
        self.crs_transformer = crs_transformer
        self.writable = writable

        self.set_grid(extent, options.cell_size)

        try:
            geojson = json.loads(file_to_str(uri))
            self.labels = load_geojson(
                geojson, crs_transformer, extent, options)
        except:
            if writable:
                self.labels = ClassificationLabels()
            else:
                raise ValueError('Could not open {}'.format(uri))

    def save(self, class_map):
        # Not implemented yet.
        pass
