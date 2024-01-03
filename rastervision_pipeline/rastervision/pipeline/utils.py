from typing import Any, Callable, Optional
import os
import atexit
import logging
from math import ceil

log = logging.getLogger(__name__)


def terminate_at_exit(process):
    def terminate():
        log.debug('Terminating {}...'.format(process.pid))
        process.terminate()

    atexit.register(terminate)


def grouped(lst: list, size: int) -> list:
    """Returns a list of lists of length 'size'.
    The last list will have size <= 'size'.
    """
    return [lst[n:n + size] for n in range(0, len(lst), size)]


def split_into_groups(lst: list, num_groups: int) -> list:
    """Attempts to split a list into a given number of groups.

    The number of groups will be at least 1 and at most
    num_groups.

    Args:
       lst: The list to split.
       num_groups: The number of groups to create.

    Returns:
       A list of size between 1 and num_groups containing lists of items of l.
    """
    group_sz = max(int(ceil((len(lst)) / num_groups)), 1)

    return grouped(lst, group_sz)


def repr_with_args(obj: Any, **kwargs) -> str:
    """Builds a string of the form: <obj's class name>(k1=v1, k2=v2, ...)."""
    cls = type(obj).__name__
    arg_strs = [f'{k}={v!r}' for k, v in kwargs.items()]
    arg_str = ', '.join(arg_strs)
    return f'{cls}({arg_str})'


def get_env_var(key: str,
                default: Optional[Any] = None,
                out_type: Optional[type | Callable] = None) -> Optional[Any]:
    val = os.environ.get(key, default)
    if val is not None and out_type is not None:
        if out_type == bool:
            return val.lower() in ('1', 'true', 'y', 'yes')
        return out_type(val)
