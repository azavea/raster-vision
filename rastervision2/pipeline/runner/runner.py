from abc import abstractmethod

class Runner():
    @abstractmethod
    def run(self, cfg_json_uri, pipeline, commands, num_splits=1):
        pass

    def get_split_ind(self):
        return None
