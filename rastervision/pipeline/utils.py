import atexit
import logging
from math import ceil

log = logging.getLogger(__name__)


def terminate_at_exit(process):
    def terminate():
        log.debug('Terminating {}...'.format(process.pid))
        process.terminate()

    atexit.register(terminate)


def grouped(lst, size):
    """Returns a list of lists of length 'size'.
    The last list will have size <= 'size'.
    """
    return [lst[n:n + size] for n in range(0, len(lst), size)]


def split_into_groups(lst, num_groups):
    """Attempts to split a list into a given number of groups.
    The number of groups will be at least 1 and at most
    num_groups.

    Args:
       lst:             The list to split
       num_groups:      The number of groups to create.
    Returns:
       A list of size between 1 and num_groups containing lists
       of items of l."""
    group_sz = max(int(ceil((len(lst)) / num_groups)), 1)

    return grouped(lst, group_sz)
