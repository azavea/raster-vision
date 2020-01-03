class Verbosity:
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    VERY_VERBOSE = 3
    DEBUG = 4

    @staticmethod
    def get():
        from rastervision.v2.core import _rv_config
        return _rv_config.get_verbosity()
