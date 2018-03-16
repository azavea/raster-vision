class TrainingData(object):
    """A set of chips and associated annotations used to train a model."""

    def __init__(self):
        """Construct a new TrainingData."""
        self.chips = []
        self.annotations = []

    def append(self, chip, annotations):
        """Append a chip and associated annotations to the dataset.

        Args:
            chip: [height, width, channels] numpy array
            annotations: Annotations
        """
        self.chips.append(chip)
        self.annotations.append(annotations)

    def __iter__(self):
        return zip(self.chips, self.annotations)
