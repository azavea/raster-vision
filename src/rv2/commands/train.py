from rv2.core.command import Command


class Train(Command):
    def __init__(self, ml_method, options):
        self.ml_method = ml_method
        self.options = options

    def run(self):
        self.ml_method.train(self.options)
