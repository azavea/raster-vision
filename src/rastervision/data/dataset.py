class Dataset:
    def __init__(self,
                 train_scenes=[],
                 validation_scenes=[],
                 test_scenes=[],
                 augmentors=[]):
        self.train_scenes = train_scenes
        self.validation_scenes = validation_scenes
        self.test_scenes = test_scenes
        self.augmentors = augmentors
