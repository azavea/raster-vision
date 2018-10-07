import io

from PIL import Image
import numpy as np
import imageio
import atexit
import logging

log = logging.getLogger(__name__)


def save_img(im_array, output_path):
    imageio.imwrite(output_path, im_array)


def numpy_to_png(array: np.ndarray) -> str:
    """Get a PNG string from a Numpy array.

    Args:
         array: A Numpy array of shape (w, h, 3) or (w, h), where the
               former is meant to become a three-channel image and the
               latter a one-channel image.  The dtype of the array
               should be uint8.

    Returns:
         str

    """
    im = Image.fromarray(array)
    output = io.BytesIO()
    im.save(output, 'png')
    return output.getvalue()


def png_to_numpy(png: str, dtype=np.uint8) -> np.ndarray:
    """Get a Numpy array from a PNG string.

    Args:
         png: A str containing a PNG-formatted image.

    Returns:
         numpy.ndarray

    """
    incoming = io.BytesIO(png)
    im = Image.open(incoming)
    return np.array(im)


def replace_nones_in_dict(target, replace_value):
    """Recursively replaces Nones in a dictionary with the given value."""
    for k in target:
        if target[k] is None:
            target[k] = replace_value
        elif type(target[k]) is list:
            result = []
            for e in target[k]:
                if type(e) is dict:
                    result.append(replace_nones_in_dict(e, replace_value))
                else:
                    if e is None:
                        result.append(replace_value)
                    else:
                        result.append(e)
            target[k] = result
        elif type(target[k]) is dict:
            replace_nones_in_dict(target[k], replace_value)
    return target


def set_nested_keys(target,
                    mods,
                    ignore_missing_keys=False,
                    set_missing_keys=False):
    """Sets dictionary keys based on modifications.

    Args:
       target - Target dictionary to be  modified in-place.
       mods - Dictionary of values to set into the target dict.
              This method will look for any keys matching the mod
              key, even in nested dictionaries. If the mod has a nested
              dictionary, then the leaf key value will only be set
              if that parent dictionary key is found and is a dictionary.
       ignore_missing_keys - If a key is not found, do not throw an error.
       set_missing_keys - If a key is not found, set it. If the key is part
                          of a nested set, and parent keys are found in the target
                          dictionary, then set the key at whatever level of the nested
                          set of keys where the key is first not found.
    """
    searched_keys, found_keys = [], []

    def f(_target, _mods, parent_key=None, mod_parent_key=None):
        for key in _target:
            if key in _mods.keys():
                found_keys.append(key)
                if type(_target[key]) is dict:
                    if type(_mods[key]) is dict:
                        f(_target[key],
                          _mods[key],
                          parent_key=key,
                          mod_parent_key=key)
                    else:
                        raise Exception('Error: cannot modify dict with value')
                else:
                    _target[key] = _mods[key]
            else:
                if type(_target[key]) is dict:
                    f(_target[key],
                      _mods,
                      parent_key=key,
                      mod_parent_key=mod_parent_key)
        searched_keys.extend(list(_mods.keys()))

        if set_missing_keys:
            for key in set(_mods.keys()) - set(found_keys):
                if not type(
                        _mods[key]) is dict and parent_key == mod_parent_key:
                    _target[key] = _mods[key]
                    found_keys.append(key)

    f(target, mods)
    if not ignore_missing_keys:
        d = set(searched_keys) - set(found_keys)
        if d:
            raise Exception('Mod keys not found in target dict: {}'.format(d))


def terminate_at_exit(process):
    def terminate():
        log.debug('Terminating {}...'.format(process.pid))
        process.terminate()

    atexit.register(terminate)
