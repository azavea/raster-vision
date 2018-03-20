from rv2.core.labels import Labels


class ClassificationLabels(Labels):
    def __init__(self):
        self.box_to_class_id = {}

    def append(self, box, class_id):
        self.box_to_class_id[box.tuple_format()] = class_id
