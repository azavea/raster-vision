class CommandIODefinition:
    """Class which contains a set of inputs and outputs for a command,
       based on the configuration.
    """

    def __init__(self,
                 input_uris=None,
                 output_uris=None,
                 missing_input_messages=None):
        if input_uris is None:
            input_uris = set([])
        if output_uris is None:
            output_uris = set([])
        if missing_input_messages is None:
            missing_input_messages = []

        self.input_uris = input_uris
        self.output_uris = output_uris

        # Messages that declare missing inputs
        self.missing_input_messages = missing_input_messages

    def merge(self, other):
        self.input_uris = self.input_uris.union(other.input_uris)
        self.output_uris = self.output_uris.union(other.output_uris)
        self.missing_input_messages = self.missing_input_messages + \
                                      other.missing_input_messages

    def add_input(self, input_uri):
        self.input_uris.add(input_uri)

    def add_inputs(self, input_uris):
        self.input_uris = self.input_uris.union(set(input_uris))

    def add_output(self, output_uri):
        self.output_uris.add(output_uri)

    def add_outputs(self, output_uris):
        self.output_uris = self.output_uris.union(set(output_uris))

    def add_missing(self, message):
        self.missing_input_messages.append(message)
