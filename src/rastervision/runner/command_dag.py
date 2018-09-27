import networkx as nx

import rastervision as rv
from rastervision.utils.files import file_exists


class CommandDAG:
    """ A directed acyclic graph of command definitions.
    """

    def __init__(self,
                 command_definitions,
                 rerun_commands=False,
                 skip_file_check=False):
        """Generates a CommandDAG from a list of CommandDefinitions

        This logic checks if there are any non-exsiting URIs that are
        not produced as outputs by some command in the set. If so,
        it raises a ConfigError stating the missing files.
        """
        # Create a set of edges, from input_uri to command_config and
        # from command_config to output_uri. Nodes for commands are their
        # index into command_definitions.

        uri_dag = nx.DiGraph()

        for idx, command_def in enumerate(command_definitions):
            uri_dag.add_node(idx)
            for input_uri in command_def.io_def.input_uris:
                uri_dag.add_edge(input_uri, idx)

            for output_uri in command_def.io_def.output_uris:
                uri_dag.add_edge(idx, output_uri)

        # Find all source input_uris, and ensure they exist.
        if not skip_file_check:
            unsolved_sources = [
                uri for uri in uri_dag.nodes
                if (type(uri) == str and len(uri_dag.in_edges(uri)) == 0)
            ]

            missing_files = []
            for uri in unsolved_sources:
                print('Ensuring file exists: {}'.format(uri))
                if not file_exists(uri):
                    missing_files.append(uri)

            if any(missing_files):
                raise rv.ConfigError(
                    'Files do not exist and are not supplied by commands:\n'
                    '\t{}\n'.format(',\b\t'.join(missing_files)))

        # If we are not rerunning, remove commands that have existing outputs.
        self.skipped_commands = []
        if not rerun_commands:
            for idx in [idx for idx in uri_dag.nodes if type(idx) == int]:
                for output_uri in [
                        edge[1] for edge in uri_dag.out_edges(idx)
                        if file_exists(edge[1])
                ]:
                    uri_dag.remove_edge(idx, output_uri)
                if len(uri_dag.out_edges(idx)) == 0:
                    self.skipped_commands.append(command_definitions[idx])
                    uri_dag.remove_node(idx)

        # Collapse the graph to create edges from command to command.
        command_id_dag = nx.DiGraph()

        for idx in [idx for idx in uri_dag.nodes if (type(idx) == int)]:
            command_id_dag.add_node(idx)
            for upstream_idx in [
                    edge2[0] for edge1 in uri_dag.in_edges(idx)
                    for edge2 in uri_dag.in_edges(edge1[0])
            ]:
                command_id_dag.add_edge(upstream_idx, idx)

        # Feed this digraph of commands to the child runner.
        self.command_definitions = command_definitions
        self.command_id_dag = command_id_dag

    def get_sorted_commands(self):
        """Return a topologically sorted list of commands configurations.

        Returns a list of command configurations that are sorted such that every
        command that depends on some other parent command appears later
        than that parent command.
        """
        return [
            self.command_definitions[idx].command_config
            for idx in self.get_sorted_command_ids()
        ]

    def get_sorted_command_ids(self):
        """Return a topologically sorted list of commands ids.

        Returns a list of command IDs that can be used to retrieve
        specific commands out of this DAG. These are sorted such that every
        command that depends on some other parent command appears later
        than that parent command.
        """
        return [idx for idx in nx.topological_sort(self.command_id_dag)]

    def get_command(self, command_id):
        """Retrieves a command configuration for the given ID"""
        return self.command_definitions[command_id].command_config

    def get_upstream_command_ids(self, command_id):
        """Returns the command ids for upstream commands for the command
        with the given id.
        """
        return list(
            map(lambda x: x[0], self.command_id_dag.in_edges(command_id)))

    def get_command_definitions(self):
        """Returns the command definitions that will be run in this DAG."""
        return [
            self.command_definitions[idx] for idx in self.command_id_dag.nodes
        ]
