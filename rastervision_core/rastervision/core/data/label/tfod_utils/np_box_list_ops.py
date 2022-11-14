# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bounding Box List operations for Numpy BoxLists."""
from typing import List, Optional, Tuple

import numpy as np

from rastervision.core.data.label.tfod_utils.np_box_list import BoxList
from rastervision.core.data.label.tfod_utils import np_box_ops


class SortOrder(object):
    """Enum class for sort order.

    Attributes:
        ascend: ascend order.
        descend: descend order.
    """
    ASCEND = 1
    DESCEND = 2


def area(boxlist: BoxList) -> np.ndarray:
    """Computes area of boxes.

    Args:
        boxlist (BoxList): BoxList holding N boxes.

    Returns:
        np.ndarray: A numpy array with shape [N*1] representing box areas.
    """
    y_min, x_min, y_max, x_max = boxlist.get_coordinates()
    return (y_max - y_min) * (x_max - x_min)


def intersection(boxlist1: BoxList, boxlist2: BoxList) -> np.ndarray:
    """Compute pairwise intersection areas between boxes.

    Args:
        boxlist1 (BoxList): BoxList holding N boxes.
        boxlist2 (BoxList): BoxList holding M boxes.

    Returns:
        np.ndarray: A numpy array with shape [N*M] representing pairwise
            intersection area.
    """
    return np_box_ops.intersection(boxlist1.get(), boxlist2.get())


def iou(boxlist1: BoxList, boxlist2: BoxList) -> np.ndarray:
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxlist1 (BoxList): BoxList holding N boxes.
        boxlist2 (BoxList): BoxList holding M boxes.

    Returns:
        np.ndarray: A numpy array with shape [N, M] representing pairwise iou scores.
    """
    return np_box_ops.iou(boxlist1.get(), boxlist2.get())


def ioa(boxlist1: BoxList, boxlist2: BoxList) -> np.ndarray:
    """Computes pairwise intersection-over-area between box collections.

    Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, IOA(box1, box2) != IOA(box2, box1).

    Args:
        boxlist1 (BoxList): BoxList holding N boxes.
        boxlist2 (BoxList): BoxList holding M boxes.

    Returns:
        np.ndarray: A numpy array with shape [N, M] representing pairwise ioa
            scores.
    """
    return np_box_ops.ioa(boxlist1.get(), boxlist2.get())


def gather(boxlist: BoxList,
           indices: np.ndarray,
           fields: Optional[List[str]] = None) -> BoxList:
    """Gather boxes from BoxList according to indices and return new BoxList.

    By default, gather returns boxes corresponding to the input index list, as
    well as all additional fields stored in the boxlist (indexing into the
    first dimension). However one can optionally only gather from a
    subset of fields.

    Args:
        boxlist (BoxList): BoxList holding N boxes.
        indices (np.ndarray): A 1-d numpy array of type int.
        fields (Optional[List[str]]): List of fields to also gather from. If
            None, all fields are gathered from. Pass an empty fields list to
            only gather the box coordinates. Defaults to None.

    Returns:
        BoxList: a BoxList corresponding to the subset of the input BoxList
            specified by indices

    Raises:
        ValueError: If specified field is not contained in boxlist or if the
            indices are not of type int.
    """
    if indices.size:
        if np.amax(indices) >= boxlist.num_boxes() or np.amin(indices) < 0:
            raise ValueError('indices are out of valid range.')
    subboxlist = BoxList(boxlist.get()[indices, :])
    if fields is None:
        fields = boxlist.get_extra_fields()
    for field in fields:
        extra_field_data = boxlist.get_field(field)
        subboxlist.add_field(field, extra_field_data[indices, ...])
    return subboxlist


def sort_by_field(boxlist: BoxList,
                  field: str,
                  order: SortOrder = SortOrder.DESCEND):
    """Sort boxes and associated fields according to a scalar field.
    A common use case is reordering the boxes according to descending scores.

    Args:
        boxlist (BoxList): A BoxList holding N boxes.
        field (str): A BoxList field for sorting and reordering the BoxList.
        order (SortOrder, optional): 'descend' or 'ascend'. Default is descend.

    Returns:
        BoxList: A sorted BoxList with the field in the specified order.

    Raises:
        ValueError: If specified field does not exist or is not of single
            dimension.
        ValueError: If the order is not either descend or ascend.
    """
    if not boxlist.has_field(field):
        raise ValueError('Field ' + field + ' does not exist')
    if len(boxlist.get_field(field).shape) != 1:
        raise ValueError('Field ' + field + 'should be single dimension.')
    if order != SortOrder.DESCEND and order != SortOrder.ASCEND:
        raise ValueError('Invalid sort order')

    field_to_sort = boxlist.get_field(field)
    sorted_indices = np.argsort(field_to_sort)
    if order == SortOrder.DESCEND:
        sorted_indices = sorted_indices[::-1]
    return gather(boxlist, sorted_indices)


def non_max_suppression(boxlist: BoxList,
                        max_output_size: int = 10_000,
                        iou_threshold: float = 1.0,
                        score_threshold: float = -10.0) -> BoxList:
    """Non maximum suppression.
    This op greedily selects a subset of detection bounding boxes, pruning
    away boxes that have high IOU (intersection over union) overlap (> thresh)
    with already selected boxes. In each iteration, the detected bounding box with
    highest score in the available pool is selected.

    Args:
        boxlist (BoxList): BoxList holding N boxes. Must contain a 'scores'
            field representing detection scores. All scores belong to the same
            class.
        max_output_size (int): Maximum number of retained boxes.
            Defaults to 10_000.
        iou_threshold (float): Intersection over union threshold.
            Defaults to 1.0.
        score_threshold (float): Minimum score threshold. Remove the boxes with
            scores less than this value. Default value is set to -10. A very
            low threshold to pass pretty much all the boxes, unless the user
            sets a different score threshold. Defaults to -10.0.

    Returns:
        BoxList: A BoxList holding M boxes. where M <= max_output_size.

    Raises:
        ValueError: If 'scores' field does not exist.
        ValueError: If threshold is not in [0, 1].
        ValueError: If max_output_size < 0.
    """
    if not boxlist.has_field('scores'):
        raise ValueError('Field scores does not exist')
    if iou_threshold < 0. or iou_threshold > 1.0:
        raise ValueError('IOU threshold must be in [0, 1]')
    if max_output_size < 0:
        raise ValueError('max_output_size must be bigger than 0.')

    boxlist = filter_scores_greater_than(boxlist, score_threshold)
    if boxlist.num_boxes() == 0:
        return boxlist

    boxlist = sort_by_field(boxlist, 'scores')

    # Prevent further computation if NMS is disabled.
    if iou_threshold == 1.0:
        if boxlist.num_boxes() > max_output_size:
            selected_indices = np.arange(max_output_size)
            return gather(boxlist, selected_indices)
        else:
            return boxlist

    boxes = boxlist.get()
    num_boxes = boxlist.num_boxes()
    # is_index_valid is True only for all remaining valid boxes,
    is_index_valid = np.full(num_boxes, 1, dtype=bool)
    selected_indices = []
    num_output = 0
    for i in range(num_boxes):
        if num_output < max_output_size:
            if is_index_valid[i]:
                num_output += 1
                selected_indices.append(i)
                is_index_valid[i] = False
                valid_indices = np.where(is_index_valid)[0]
                if valid_indices.size == 0:
                    break

                intersect_over_union = np_box_ops.iou(
                    np.expand_dims(boxes[i, :], axis=0),
                    boxes[valid_indices, :])
                intersect_over_union = np.squeeze(intersect_over_union, axis=0)
                is_index_valid[valid_indices] = np.logical_and(
                    is_index_valid[valid_indices],
                    intersect_over_union <= iou_threshold)
    return gather(boxlist, np.array(selected_indices))


def multi_class_non_max_suppression(boxlist: BoxList, score_thresh: float,
                                    iou_thresh: float,
                                    max_output_size: int) -> BoxList:
    """Multi-class version of non maximum suppression.

    This op greedily selects a subset of detection bounding boxes, pruning
    away boxes that have high IOU (intersection over union) overlap (> thresh)
    with already selected boxes. It operates independently for each class for
    which scores are provided (via the scores field of the input box_list),
    pruning boxes with score less than a provided threshold prior to
    applying NMS.

    Args:
        boxlist (BoxList): A BoxList holding N boxes. Must contain a 'scores' field
            representing detection scores. This scores field is a tensor that
            can be 1 dimensional (in the case of a single class) or
            2-dimensional, which which case we assume that it takes the shape
            [num_boxes, num_classes]. We further assume that this rank is known
            statically and that scores.shape[1] is also known (i.e., the number
            of classes is fixed and known at graph construction time).
        score_thresh (float): Scalar threshold for score (low scoring boxes are
            removed).
        iou_thresh (float): Scalar threshold for IOU (boxes that that high IOU
            overlap with previously selected boxes are removed).
        max_output_size (int): maximum number of retained boxes per class.

    Returns:
        BoxList: BoxList with M boxes with a rank-1 scores field representing
            corresponding scores for each box with scores sorted in decreasing
            order and a rank-1 classes field representing a class label for
            each box.

    Raises:
        ValueError: If iou_thresh is not in [0, 1] or if input boxlist does not
            have a valid scores field.
    """
    if not 0 <= iou_thresh <= 1.0:
        raise ValueError('thresh must be between 0 and 1')
    if not isinstance(boxlist, BoxList):
        raise ValueError('boxlist must be a BoxList')
    if not boxlist.has_field('scores'):
        raise ValueError('input boxlist must have \'scores\' field')
    scores = boxlist.get_field('scores')
    if len(scores.shape) == 1:
        scores = np.reshape(scores, [-1, 1])
    elif len(scores.shape) == 2:
        if scores.shape[1] is None:
            raise ValueError(
                'scores field must have statically defined second '
                'dimension')
    else:
        raise ValueError('scores field must be of rank 1 or 2')
    num_boxes = boxlist.num_boxes()
    num_scores = scores.shape[0]
    num_classes = scores.shape[1]

    if num_boxes != num_scores:
        raise ValueError('Incorrect scores field length: actual vs expected.')

    selected_boxes_list = []
    for class_idx in range(num_classes):
        boxlist_and_class_scores = BoxList(boxlist.get())
        class_scores = np.reshape(scores[0:num_scores, class_idx], [-1])
        boxlist_and_class_scores.add_field('scores', class_scores)
        boxlist_filt = filter_scores_greater_than(boxlist_and_class_scores,
                                                  score_thresh)
        nms_result = non_max_suppression(
            boxlist_filt,
            max_output_size=max_output_size,
            iou_threshold=iou_thresh,
            score_threshold=score_thresh)
        nms_result.add_field(
            'classes',
            np.zeros_like(nms_result.get_field('scores')) + class_idx)
        selected_boxes_list.append(nms_result)
    selected_boxes = concatenate(selected_boxes_list)
    sorted_boxes = sort_by_field(selected_boxes, 'scores')
    return sorted_boxes


def scale(boxlist: BoxList, y_scale: float, x_scale: float) -> BoxList:
    """Scale box coordinates in x and y dimensions.

    Args:
        boxlist (BoxList): A BoxList holding N boxes.
        y_scale (float):
        x_scale (float):

    Returns:
        BoxList: A BoxList holding N boxes.
    """
    y_min, x_min, y_max, x_max = np.array_split(boxlist.get(), 4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = BoxList(np.hstack([y_min, x_min, y_max, x_max]))

    fields = boxlist.get_extra_fields()
    for field in fields:
        extra_field_data = boxlist.get_field(field)
        scaled_boxlist.add_field(field, extra_field_data)

    return scaled_boxlist


def clip_to_window(boxlist: BoxList, window: np.ndarray) -> BoxList:
    """Clip bounding boxes to a window.

    This op clips input bounding boxes (represented by bounding box
    corners) to a window, optionally filtering out boxes that do not
    overlap at all with the window.

    Args:
        boxlist (BoxList): A BoxList holding M_in boxes
        window: a numpy array of shape [4] representing the
            [y_min, x_min, y_max, x_max] window to which the op
            should clip boxes.

    Returns:
        BoxList: A BoxList holding M_out boxes where M_out <= M_in
    """
    y_min, x_min, y_max, x_max = np.array_split(boxlist.get(), 4, axis=1)
    win_y_min = window[0]
    win_x_min = window[1]
    win_y_max = window[2]
    win_x_max = window[3]
    y_min_clipped = np.fmax(np.fmin(y_min, win_y_max), win_y_min)
    y_max_clipped = np.fmax(np.fmin(y_max, win_y_max), win_y_min)
    x_min_clipped = np.fmax(np.fmin(x_min, win_x_max), win_x_min)
    x_max_clipped = np.fmax(np.fmin(x_max, win_x_max), win_x_min)
    clipped = BoxList(
        np.hstack([y_min_clipped, x_min_clipped, y_max_clipped,
                   x_max_clipped]))
    clipped = _copy_extra_fields(clipped, boxlist)
    areas = area(clipped)
    nonzero_area_indices = np.reshape(
        np.nonzero(np.greater(areas, 0.0)), [-1]).astype(np.int32)
    return gather(clipped, nonzero_area_indices)


def prune_non_overlapping_boxes(boxlist1: BoxList,
                                boxlist2: BoxList,
                                minoverlap: float = 0.0) -> BoxList:
    """Prunes boxes with insufficient overlap b/w boxlists.

    Prunes the boxes in boxlist1 that overlap less than thresh with boxlist2.
    For each box in boxlist1, we want its IOA to be more than minoverlap with
    at least one of the boxes in boxlist2. If it does not, we remove it.

    Args:
        boxlist1 (BoxList): BoxList holding N boxes.
        boxlist2 (BoxList): BoxList holding M boxes.
        minoverlap: float: Minimum required overlap between boxes, to count
            them as overlapping. Defaults to 0.0.

    Returns:
        BoxList: A pruned boxlist with size [N', 4].
    """
    intersection_over_area = ioa(boxlist2, boxlist1)  # [M, N] tensor
    intersection_over_area = np.amax(
        intersection_over_area, axis=0)  # [N] tensor
    keep_bool = np.greater_equal(intersection_over_area, np.array(minoverlap))
    keep_inds = np.nonzero(keep_bool)[0]
    new_boxlist1 = gather(boxlist1, keep_inds)
    return new_boxlist1


def prune_outside_window(boxlist: BoxList,
                         window: np.ndarray) -> Tuple[BoxList, np.ndarray]:
    """Prunes bounding boxes that fall outside a given window.

    This function prunes bounding boxes that even partially fall outside the
    given window. See also ClipToWindow which only prunes bounding boxes that
    fall completely outside the window, and clips any bounding boxes that
    partially overflow.

    Args:
        boxlist (BoxList): A BoxList holding M_in boxes.
        window (np.ndaarray): A numpy array of size 4, representing
            [ymin, xmin, ymax, xmax] of the window.

    Returns:
        Tuple[BoxList, np.ndarray]: Pruned Boxlist of length <= M_in and
            an array of shape [M_out] indexing the valid bounding boxes in the
            input tensor.
    """

    y_min, x_min, y_max, x_max = np.array_split(boxlist.get(), 4, axis=1)
    win_y_min = window[0]
    win_x_min = window[1]
    win_y_max = window[2]
    win_x_max = window[3]
    coordinate_violations = np.hstack([
        np.less(y_min, win_y_min),
        np.less(x_min, win_x_min),
        np.greater(y_max, win_y_max),
        np.greater(x_max, win_x_max)
    ])
    valid_indices = np.reshape(
        np.where(np.logical_not(np.max(coordinate_violations, axis=1))), [-1])
    pruned_boxlist = gather(boxlist, valid_indices)
    return pruned_boxlist, valid_indices


def concatenate(boxlists: List[BoxList],
                fields: Optional[List[str]] = None) -> BoxList:
    """Concatenate list of BoxLists.

    This op concatenates a list of input BoxLists into a larger BoxList. It also
    handles concatenation of BoxList fields as long as the field tensor shapes
    are equal except for the first dimension.

    Args:
        boxlists (List[BoxList]): List of BoxList objects.
        fields (Optional[List[str]]): Optional list of fields to also
            concatenate. If None, all fields from the first BoxList in the
            list are included in the concatenation. Defaults to None.

    Returns:
        BoxList: A BoxList with number of boxes equal to
            sum([boxlist.num_boxes() for boxlist in BoxList])

    Raises:
        ValueError: If boxlists is invalid (i.e., is not a list, is empty, or
            contains non BoxList objects), or if requested fields are not contained
            in all boxlists
    """
    if not isinstance(boxlists, list):
        raise ValueError('boxlists should be a list')
    if not boxlists:
        raise ValueError('boxlists should have nonzero length')
    for boxlist in boxlists:
        if not isinstance(boxlist, BoxList):
            raise ValueError(
                'all elements of boxlists should be BoxList objects')
    concatenated = BoxList(np.vstack([boxlist.get() for boxlist in boxlists]))
    if fields is None:
        fields = boxlists[0].get_extra_fields()
    for field in fields:
        first_field_shape = boxlists[0].get_field(field).shape
        first_field_shape = first_field_shape[1:]
        for boxlist in boxlists:
            if not boxlist.has_field(field):
                raise ValueError('boxlist must contain all requested fields')
            field_shape = boxlist.get_field(field).shape
            field_shape = field_shape[1:]
            if field_shape != first_field_shape:
                raise ValueError(
                    'field %s must have same shape for all boxlists '
                    'except for the 0th dimension.' % field)
        concatenated_field = np.concatenate(
            [boxlist.get_field(field) for boxlist in boxlists], axis=0)
        concatenated.add_field(field, concatenated_field)
    return concatenated


def filter_scores_greater_than(boxlist: BoxList, thresh: float) -> BoxList:
    """Filter to keep only boxes with score exceeding a given threshold.

    This op keeps the collection of boxes whose corresponding scores are
    greater than the input threshold.

    Args:
        boxlist (BoxList): A BoxList holding N boxes. Must contain a 'scores'
            field representing detection scores.
        thresh (float): scalar threshold

    Returns:
        BoxList: A BoxList holding M boxes. where M <= N

    Raises:
        ValueError: If boxlist not a BoxList object or if it does not have a
            scores field.
    """
    if not isinstance(boxlist, BoxList):
        raise ValueError('boxlist must be a BoxList')
    if not boxlist.has_field('scores'):
        raise ValueError('input boxlist must have \'scores\' field')
    scores = boxlist.get_field('scores')
    if len(scores.shape) > 2:
        raise ValueError('Scores should have rank 1 or 2')
    if len(scores.shape) == 2 and scores.shape[1] != 1:
        raise ValueError('Scores should have rank 1 or have shape '
                         'consistent with [None, 1]')
    high_score_indices = np.reshape(
        np.where(np.greater(scores, thresh)), [-1]).astype(np.int32)
    return gather(boxlist, high_score_indices)


def change_coordinate_frame(boxlist: BoxList, window: np.ndarray) -> BoxList:
    """Change coordinate frame of the boxlist to be relative to window's frame.

    Given a window of the form [ymin, xmin, ymax, xmax],
    changes bounding box coordinates from boxlist to be relative to this window
    (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).
    An example use case is data augmentation: where we are given groundtruth
    boxes (boxlist) and would like to randomly crop the image to some
    window (window). In this case we need to change the coordinate frame of
    each groundtruth box to be relative to this new window.

    Args:
        boxlist: A BoxList object holding N boxes.
        window: a size 4 1-D numpy array.

    Returns:
        BoxList: Returns a BoxList object with N boxes.
    """
    win_height = window[2] - window[0]
    win_width = window[3] - window[1]
    boxlist_new = scale(
        BoxList(boxlist.get() - [window[0], window[1], window[0], window[1]]),
        1.0 / win_height, 1.0 / win_width)
    _copy_extra_fields(boxlist_new, boxlist)

    return boxlist_new


def _copy_extra_fields(boxlist_to_copy_to: BoxList,
                       boxlist_to_copy_from: BoxList) -> BoxList:
    """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.

    Args:
        boxlist_to_copy_to (BoxList): BoxList to which extra fields are copied.
        boxlist_to_copy_from (BoxList): BoxList from which fields are copied.

    Returns:
        BoxList: boxlist_to_copy_to with extra fields.
    """
    for field in boxlist_to_copy_from.get_extra_fields():
        boxlist_to_copy_to.add_field(field,
                                     boxlist_to_copy_from.get_field(field))
    return boxlist_to_copy_to
