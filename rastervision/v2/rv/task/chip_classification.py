from rastervision.v2.rv.task import Task


class ChipClassification(Task):
    def analyze(self, split_ind=0, num_splits=1):
        pass

    def chip(self, split_ind=0, num_splits=1):
        print(self.config)

    def train(self):
        pass

    def predict(self, split_ind=0, num_splits=1):
        pass

    def eval(self):
        pass
