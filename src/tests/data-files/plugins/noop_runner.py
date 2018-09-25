from rastervision.runner import (ExperimentRunner, CommandRunner)

NOOP_RUNNER = 'NOOP_RUNNER'


class NoopExperimentRunner(ExperimentRunner):
    def _run_experiment(self, command_dag):
        for command_config in command_dag.get_sorted_commands():
            msg = command_config.to_proto()
            CommandRunner.run_from_proto(msg)


def register_plugin(plugin_registry):
    plugin_registry.register_experiment_runner(NOOP_RUNNER,
                                               NoopExperimentRunner)
