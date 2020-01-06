from rastervision.v2.core.pipeline import Pipeline


class Task(Pipeline):
    commands = ['analyze', 'chip', 'train', 'predict', 'eval']
    split_commands = ['analyze', 'chip', 'predict']
    gpu_commands = ['train', 'predict']

    def analyze(self, split_ind=0, num_splits=1):
        pass

    def chip(self, split_ind=0, num_splits=1):
        pass

    def train(self):
        pass

    def predict(self, split_ind=0, num_splits=1):
        pass

    def eval(self):
        pass
