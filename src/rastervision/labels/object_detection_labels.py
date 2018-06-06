import numpy as np

from object_detection.utils.np_box_list import BoxList
from object_detection.utils.np_box_list_ops import (
    prune_non_overlapping_boxes, clip_to_window, change_coordinate_frame,
    concatenate, scale, multi_class_non_max_suppression, _copy_extra_fields)

from rastervision.core.box import Box
from rastervision.core.labels import Labels
from rastervision.labels.utils import boxes_to_geojson


def geojson_to_labels(geojson, crs_transformer, extent):
    """Extract boxes and related info from GeoJSON file."""
    features = geojson['features']
    boxes = []
    class_ids = []
    scores = []

    for feature in features:
        # Convert polygon to pixel coords and then convert to bounding box.
        polygon = feature['geometry']['coordinates'][0]
        polygon = [crs_transformer.web_to_pixel(p) for p in polygon]
        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)
        boxes.append(Box(ymin, xmin, ymax, xmax))

        properties = feature['properties']
        class_ids.append(properties['class_id'])
        scores.append(properties.get('score', 1.0))

    boxes = np.array([box.npbox_format() for box in boxes], dtype=float)
    class_ids = np.array(class_ids)
    scores = np.array(scores)
    labels = ObjectDetectionLabels(boxes, class_ids, scores=scores)
    labels = labels.get_intersection(extent)
    return labels


def inverse_change_coordinate_frame(boxlist, window):
    scaled_boxlist = scale(boxlist, window.get_height(), window.get_width())
    npboxes = np.round(scaled_boxlist.get())
    npboxes += [window.ymin, window.xmin, window.ymin, window.xmin]
    boxlist_new = BoxList(npboxes)
    _copy_extra_fields(boxlist_new, boxlist)
    return boxlist_new


class ObjectDetectionLabels(Labels):
    def __init__(self, npboxes, class_ids, scores=None):
        self.boxlist = BoxList(npboxes)
        # This field name actually needs to be 'classes' to be able to use
        # certain utility functions in the TF Object Detection API.
        self.boxlist.add_field('classes', class_ids)
        if scores is not None:
            self.boxlist.add_field('scores', scores)

    @staticmethod
    def from_boxlist(boxlist):
        scores = boxlist.get_field('scores') \
                 if boxlist.has_field('scores') else None
        return ObjectDetectionLabels(
            boxlist.get(), boxlist.get_field('classes'), scores)

    @staticmethod
    def from_geojson(geojson, crs_transformer, extent):
        return geojson_to_labels(geojson, crs_transformer, extent)

    @staticmethod
    def make_empty():
        npboxes = np.empty((0, 4))
        labels = np.empty((0,))
        scores = np.empty((0,))
        return ObjectDetectionLabels(npboxes, labels, scores)

    def get_subwindow(self, window, ioa_thresh=1.0):
        """Returns boxes relative to window.

        This returns the boxes that overlap enough with window, clipped to
        the window and in relative coordinates that lie between 0 and 1.
        """
        window_npbox = window.npbox_format()
        window_boxlist = BoxList(np.expand_dims(window_npbox, axis=0))
        boxlist = prune_non_overlapping_boxes(
            self.boxlist, window_boxlist, minoverlap=ioa_thresh)
        boxlist = clip_to_window(boxlist, window_npbox)
        boxlist = change_coordinate_frame(boxlist, window_npbox)
        return ObjectDetectionLabels.from_boxlist(boxlist)

    def get_boxes(self):
        return [Box.from_npbox(npbox) for npbox in self.boxlist.get()]

    def get_intersection(self, window, min_ioa=0.000001):
        """Returns list of boxes that intersect with window.

        Does not clip or perform coordinate transform.
        """
        window_npbox = window.npbox_format()
        window_boxlist = BoxList(np.expand_dims(window_npbox, axis=0))
        boxlist = prune_non_overlapping_boxes(
            self.boxlist, window_boxlist, minoverlap=min_ioa)
        return ObjectDetectionLabels.from_boxlist(boxlist)

    def get_coordinates(self):
        return self.boxlist.get_coordinates()

    def get_npboxes(self):
        return self.boxlist.get()

    def get_scores(self):
        if self.boxlist.has_field('scores'):
            return self.boxlist.get_field('scores')
        return None

    def get_class_ids(self):
        return self.boxlist.get_field('classes')

    def __len__(self):
        return self.boxlist.get().shape[0]

    def __str__(self):
        return str(self.boxlist.get())

    def concatenate(self, window, labels):
        boxlist_new = concatenate([
            self.boxlist,
            inverse_change_coordinate_frame(labels.boxlist, window)])
        return ObjectDetectionLabels.from_boxlist(boxlist_new)

    def prune_duplicates(self, score_thresh, merge_thresh):
        max_output_size = 1000000

        # Create a copy of self.boxlist that has a 2D scores
        # field with a column for each class which is required
        # by the multi_class_non_max_suppression function. It's
        # suprising that the scores field has to be in this form since
        # I haven't seen other functions require that.
        boxlist = BoxList(self.boxlist.get())
        classes = self.boxlist.get_field('classes').astype(np.int32)
        nb_boxes = classes.shape[0]
        nb_classes = np.max(classes)
        class_inds = classes - 1
        scores_1d = self.boxlist.get_field('scores')
        scores_2d = np.zeros((nb_boxes, nb_classes))
        # Not sure how to vectorize this so just do for loop :(
        for box_ind in range(nb_boxes):
            scores_2d[box_ind, class_inds[box_ind]] = scores_1d[box_ind]
        boxlist.add_field('scores', scores_2d)

        pruned_boxlist = multi_class_non_max_suppression(
            boxlist, score_thresh, merge_thresh, max_output_size)
        # Add one because multi_class_nms outputs labels that start at zero
        # instead of one like in the rest of the system.
        class_ids = pruned_boxlist.get_field('classes')
        class_ids += 1
        return ObjectDetectionLabels.from_boxlist(pruned_boxlist)

    def to_geojson(self, crs_transformer, class_map):
        boxes = self.get_boxes()
        class_ids = self.get_class_ids().tolist()
        scores = self.get_scores().tolist()

        return boxes_to_geojson(boxes, class_ids, crs_transformer, class_map,
                                scores=scores)
