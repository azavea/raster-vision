class Verbosity:
    """Verbosity level for the sake of logging."""
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    VERY_VERBOSE = 3
    DEBUG = 4

    @staticmethod
    def get() -> 'rastervision2.pipeline.Verbosity':
        """Get the verbosity from RVConfig."""
        from rastervision2.pipeline import rv_config
        return rv_config.get_verbosity()
