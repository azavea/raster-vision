import sys

from rastervision.v2.core.cli.main import main

# Code taken from unittest
if sys.argv[0].endswith('__main__.py'):
    import os.path
    # We change sys.argv[0] to make help message more useful
    # use executable without path, unquoted
    # (it's just a hint anyway)
    # (if you have spaces in your executable you get what you deserve!)
    executable = os.path.basename(sys.executable)
    sys.argv[0] = executable + ' -m rastervision.v2'
    del os

main()
