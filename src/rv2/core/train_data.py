class TrainData(object):
    def __init__(self):
        self.chips = []
        self.annotations = []

    def append(self, chip, annotations):
        self.chips.append(chip)
        self.annotations.append(annotations)
