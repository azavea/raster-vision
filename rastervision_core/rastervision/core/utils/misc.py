from pydantic import confloat

import imageio
import logging

Proportion = confloat(ge=0, le=1)

log = logging.getLogger(__name__)


def save_img(im_array, output_path):
    imageio.imwrite(output_path, im_array)
