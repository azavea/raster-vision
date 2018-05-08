import random


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

    def shuffle(self):
        """Randomly shuffle the chips and labels in-place.

        This maintains the correspondence between chips and labels.
        """
        chip_labels = list(self)
        random.shuffle(chip_labels)
        # Unzip the list.
        self.chips, self.labels = zip(*chip_labels)
