from rastervision.command import Command


class AnalyzeCommand(Command):
    def __init__(self, scenes, analyzers):
        self.scenes = scenes
        self.analyzers = analyzers

    def run(self, tmp_dir):
        for analyzer in self.analyzers:
            analyzer.process(self.scenes, tmp_dir)
