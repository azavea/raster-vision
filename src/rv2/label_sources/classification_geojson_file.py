import json

import numpy as np

from rv2.labels.classification_labels import (
    ClassificationLabels)
from rv2.labels.object_detection_labels import ObjectDetectionLabels
from rv2.labels.utils import boxes_to_geojson
from rv2.utils.files import file_to_str, str_to_file
from rv2.label_sources.classification_label_source import (
        ClassificationLabelSource)


def infer_labels(od_labels, extent, options):
    labels = ClassificationLabels()
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


def convert_labels(od_labels, extent, options):
    labels = ClassificationLabels()
    boxes = od_labels.get_boxes()
    class_ids = od_labels.get_class_ids()

    for box, class_id in zip(boxes, class_ids):
        labels.set_cell(box, class_id)

    return labels


def load_geojson(geojson, crs_transformer, extent, options):
    """Construct ClassificationLabels from GeoJSON.

    Args:
        options: ClassificationGeoJSONFile.Options
    """
    # Use the ObjectDetectionLabels to parse bounding boxes out of the GeoJSON
    # which contains polygons and infer which boxes lie within cells.
    od_labels = ObjectDetectionLabels.from_geojson(
        geojson, crs_transformer)

    if options.infer_cells:
        labels = infer_labels(od_labels, extent, options)
    else:
        labels = convert_labels(od_labels, extent, options)

    return labels


def to_geojson(labels, crs_transformer, class_map):
    boxes = labels.get_cells()
    class_ids = labels.get_class_ids()

    return boxes_to_geojson(
        boxes, class_ids, crs_transformer, class_map)


class ClassificationGeoJSONFile(ClassificationLabelSource):
    """A GeoJSON file with classification labels in it.

    Ideally the GeoJSON file contains a square for each cell in the grid. But
    in reality, it can be difficult to label imagery in such an exhaustive way.
    So, this can also handle GeoJSON files with non-overlapping polygons that
    do not necessarily cover the entire extent. It infers the grid of cells
    and associated class_ids using the extent and options if infer_cells is
    set to True.

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
        except:  # TODO do a better job of only catching "not found" errors
            if writable:
                self.labels = ClassificationLabels()
            else:
                raise ValueError('Could not open {}'.format(uri))

    def save(self, class_map):
        if self.writable:
            geojson = to_geojson(
                self.labels, self.crs_transformer, class_map)
            geojson_str = json.dumps(geojson)
            str_to_file(geojson_str, self.uri)
        else:
            raise ValueError('Cannot save with writable=False')
