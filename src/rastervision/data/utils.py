from typing import Tuple

from PIL import ImageColor
import numpy as np


def color_to_triple(color: str) -> Tuple[int, int, int]:
    """Given a PIL ImageColor string, return a triple of integers
    representing the red, green, and blue values.

    Args:
         color: A PIL ImageColor string

    Returns:
         An triple of integers

    """
    if color is None:
        r = np.random.randint(0, 0x100)
        g = np.random.randint(0, 0x100)
        b = np.random.randint(0, 0x100)
        return (r, g, b)
    else:
        return ImageColor.getrgb(color)


def color_to_integer(color: str) -> int:
    """Given a PIL ImageColor string, return a packed integer.

    Args:
         color: A PIL ImageColor string

    Returns:
         An integer containing the packed RGB values.

    """
    triple = color_to_triple(color)
    r = triple[0] * (1 << 16)
    g = triple[1] * (1 << 8)
    b = triple[2] * (1 << 0)
    integer = r + g + b
    return integer


def rgb_to_int_array(rgb_array):
    r = np.array(rgb_array[:, :, 0], dtype=np.uint32) * (1 << 16)
    g = np.array(rgb_array[:, :, 1], dtype=np.uint32) * (1 << 8)
    b = np.array(rgb_array[:, :, 2], dtype=np.uint32) * (1 << 0)
    return r + g + b


class SegmentationClassTransformer():
    def __init__(self, class_map):
        color_to_class = dict(
            [(item.color, item.id) for item in class_map.get_items()])

        # color int to class
        color_int_to_class = dict(
            zip([color_to_integer(c) for c in color_to_class.keys()],
                color_to_class.values()))

        def color_int_to_class_fn(color: int) -> int:
            return color_int_to_class.get(color, 0x00)

        self.transform_color_int_to_class = \
            np.vectorize(color_int_to_class_fn, otypes=[np.uint8])

        # class to color triple
        class_to_color_triple = dict(
            zip(color_to_class.values(),
                [color_to_triple(c) for c in color_to_class.keys()]))

        def class_to_channel_color(channel: int, class_id: int) -> int:
            """Given a channel (red, green, or blue) and a class, return the
            intensity of that channel.

            Args:
                 channel: An integer with value 0, 1, or 2
                      representing the channel.
                 class_id: The class id represented as an integer.
            Returns:
                 The intensity of the channel for the color associated
                      with the given class.
            """
            default_triple = (0x00, 0x00, 0x00)
            return class_to_color_triple.get(class_id, default_triple)[channel]

        class_to_r = np.vectorize(
            lambda c: class_to_channel_color(0, c), otypes=[np.uint8])
        class_to_g = np.vectorize(
            lambda c: class_to_channel_color(1, c), otypes=[np.uint8])
        class_to_b = np.vectorize(
            lambda c: class_to_channel_color(2, c), otypes=[np.uint8])
        self.transform_class_to_color = [class_to_r, class_to_g, class_to_b]

    def rgb_to_class(self, rgb_labels):
        color_int_labels = rgb_to_int_array(rgb_labels)
        class_labels = self.transform_color_int_to_class(color_int_labels)
        return class_labels.astype(np.uint8)

    def class_to_rgb(self, class_labels):
        rgb_labels = np.empty(class_labels.shape + (3, ))
        for chan in range(3):
            class_to_channel_color = self.transform_class_to_color[chan]
            rgb_labels[:, :, chan] = class_to_channel_color(class_labels)
        return rgb_labels.astype(np.uint8)
