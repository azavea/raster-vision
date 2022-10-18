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
"""Numpy BoxList classes and functions."""

from typing import List, Tuple

import numpy as np


class BoxList(object):
    """A list of bounding boxes as a [y_min, x_min, y_max, x_max] numpy array.

    It is assumed that all bounding boxes within a given list correspond to a
    single image. Optionally, users can add additional related fields (such as
    objectness/classification scores).
    """

    def __init__(self, data: np.ndarray):
        """Constructor.

        Args:
            data (np.ndarray): Box coords as a [N, 4] numpy array.

        Raises:
            ValueError: If bbox data is not a numpy array.
            ValueError: If invalid dimensions for bbox data.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be a numpy array.')
        if len(data.shape) != 2 or data.shape[1] != 4:
            raise ValueError('Invalid dimensions for box data.')
        if data.dtype != np.float32 and data.dtype != np.float64:
            raise ValueError(
                'Invalid data type for box data: float is required.')
        if not self._is_valid_boxes(data):
            raise ValueError('Invalid box data. data must be a numpy array of '
                             'N*[y_min, x_min, y_max, x_max]')
        self.data = {'boxes': data}

    def num_boxes(self) -> int:
        """Return number of boxes held in collections."""
        return self.data['boxes'].shape[0]

    def get_extra_fields(self) -> List[str]:
        """Return all non-box fields."""
        return [k for k in self.data.keys() if k != 'boxes']

    def has_field(self, field) -> bool:
        return field in self.data

    def add_field(self, name: str, data: np.ndarray) -> None:
        """Add data to a specified field.

        Args:
            name (str): Field name.
            data (np.ndarray): Field data: box coords as a [N, 4] numpy array.

        Raises:
            ValueError: If name already exists.
            ValueError: If the dimension of the field data does not matche the
                number of boxes.
        """
        if self.has_field(name):
            raise ValueError('Field ' + name + 'already exists')

        if len(data.shape) < 1 or len(data) != self.num_boxes():
            raise ValueError('Invalid dimensions for name data')

        self.data[name] = data

    def get(self):
        """Shorthand for get_field('boxes')."""
        return self.get_field('boxes')

    def get_field(self, name: str) -> np.ndarray:
        """Get data for field.

        Args:
            name (str): Field name.

        Returns:
            np.ndarray: The data associated with the field.

        Raises:
            ValueError: if invalid field.
        """
        try:
            return self.data[name]
        except KeyError:
            raise ValueError(f'field {name} does not exist')

    def get_coordinates(
            self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get corner coordinates of boxes.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: a 4-tuple
                of 1-d numpy arrays [y_min, x_min, y_max, x_max].
        """
        boxes = self.get_field('boxes')
        return tuple(boxes.T)

    def _is_valid_boxes(self, data: np.ndarray) -> bool:
        """Check whether data fullfills the format of N*[ymin, xmin, ymax, xmin].

        Args:
            data (np.ndarray): Box coords as a [N, 4] numpy array.

        Returns:
            bool: Whether ymin <= ymax and xmin <= xmax.
        """
        ymin, xmin, ymax, xmax = data.T
        return (ymin <= ymax).all() and (xmin <= xmax).all()
