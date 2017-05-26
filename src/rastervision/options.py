from rastervision.semseg.options import SemsegOptions
from rastervision.semseg.settings import SEMSEG

from rastervision.tagging.options import TaggingOptions
from rastervision.tagging.settings import TAGGING


def make_options(options_dict):
    problem_type = options_dict.get('problem_type')

    options = None
    if problem_type is None:
        raise ValueError('problem_type must be specified in options file')
    elif problem_type == SEMSEG:
        options = SemsegOptions(options_dict)
    elif problem_type == TAGGING:
        options = TaggingOptions(options_dict)

    return options
