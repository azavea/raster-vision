import rastervision as rv


class Verbosity:
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    VERY_VERBOSE = 3
    DEBUG = 4

    @staticmethod
    def get():
        return rv._registry._get_rv_config().get_verbosity()
