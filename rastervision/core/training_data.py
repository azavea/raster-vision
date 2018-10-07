import random


class TrainingData(object):
    """A set of chips, windows, and labels used to train a model."""

    def __init__(self):
        """Construct a new TrainingData."""
        self.chips = []
        self.windows = []
        self.labels = []

    def append(self, chip, window, labels):
        """Append a chip and associated labels to the dataset.

        Args:
            chip: [height, width, channels] numpy array
            window: Box with coordinates of chip
            labels: Labels
        """
        self.chips.append(chip)
        self.windows.append(window)
        self.labels.append(labels)

    def __iter__(self):
        return zip(self.chips, self.windows, self.labels)

    def shuffle(self):
        """Randomly shuffle the chips and labels in-place.

        This maintains the correspondence between chips and labels.
        """
        chip_windows_labels = list(self)
        random.shuffle(chip_windows_labels)
        # Unzip the list.
        self.chips, self.windows, self.labels = zip(*chip_windows_labels)
