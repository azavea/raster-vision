class TrainData(object):
    """A set of chips and associated annotations used to train a model."""
    def __init__(self):
        self.chips = []
        self.annotations = []

    def append(self, chip, annotations):
        self.chips.append(chip)
        self.annotations.append(annotations)
