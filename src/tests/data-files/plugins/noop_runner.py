import rastervision as rv
from rastervision.runner import ExperimentRunner

NOOP_RUNNER = 'NOOP_RUNNER'


class NoopExperimentRunner(ExperimentRunner):
    def _run_experiment(self, command_dag, run_options):
        for command_config in command_dag.get_sorted_commands():
            msg = command_config.to_proto()
            cc = rv.command.CommandConfig.from_proto(msg)
            command = cc.create_command('NONE')
            command.run('NONE')


def register_plugin(plugin_registry):
    plugin_registry.register_experiment_runner(NOOP_RUNNER,
                                               NoopExperimentRunner)
