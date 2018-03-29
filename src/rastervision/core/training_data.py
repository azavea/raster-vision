class TrainingData(object):
    """A set of chips and associated labels used to train a model."""

    def __init__(self):
        """Construct a new TrainingData."""
        self.chips = []
        self.labels = []

    def append(self, chip, labels):
        """Append a chip and associated labels to the dataset.

        Args:
            chip: [height, width, channels] numpy array
            labels: Labels
        """
        self.chips.append(chip)
        self.labels.append(labels)

    def __iter__(self):
        return zip(self.chips, self.labels)
