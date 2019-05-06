# flake8: noqa

import rastervision as rv

ANALYZE = 'ANALYZE'
CHIP = 'CHIP'
TRAIN = 'TRAIN'
PREDICT = 'PREDICT'
EVAL = 'EVAL'
BUNDLE = 'BUNDLE'

from .command_config import CommandConfig


def all_commands():
    return rv._registry.get_commands()
