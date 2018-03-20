from rv2.core.label_source import LabelSource


class ClassificationLabelSource(LabelSource):
    def get_labels(self, window):
        pass

    def get_all_labels(self):
        pass

    def extend(self, window, labels):
        pass

    def post_process(self):
        pass

    def save(self, class_map):
        pass

    def clear(self):
        pass
