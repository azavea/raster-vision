class Dataset:
    """Comprises the scenes for train, valid, and test splits."""

    def __init__(self, train_scenes=[], validation_scenes=[], test_scenes=[]):
        self.train_scenes = train_scenes
        self.validation_scenes = validation_scenes
        self.test_scenes = test_scenes
