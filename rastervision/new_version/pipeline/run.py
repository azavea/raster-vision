import importlib
from os.path import join

import click

from rastervision.new_version.pipeline.registry import registry
from rastervision.utils.files import json_to_file


@click.command()
@click.argument('runner')
@click.argument('cfg_path')
@click.argument('commands', nargs=-1)
@click.option(
    '--arg', '-a', type=(str, str), multiple=True, metavar='KEY VALUE')
@click.option('--splits', '-s', default=1)
def run(runner, cfg_path, commands, arg, splits):
    cfg_module = importlib.import_module(cfg_path)
    args = dict(arg)

    get_config = getattr(cfg_module, 'get_config', None)
    get_configs = get_config
    if get_config is None:
        get_configs = getattr(cfg_module, 'get_configs', None)

    new_args = {}
    for k, v in args.items():
        if v.lower() == 'true':
            v = True
        elif v.lower() == 'false':
            v = False
        new_args[k] = v
    args = new_args

    cfgs = get_configs(runner, **args)
    if not isinstance(cfgs, list):
        cfgs = [cfgs]

    for cfg in cfgs:
        cfg.update_all()
        cfg_dict = cfg.dict()
        cfg_json_uri = join(cfg.root_uri, 'pipeline.json')
        json_to_file(cfg_dict, cfg_json_uri)

        pipeline = cfg.get_pipeline()
        if not commands:
            commands = pipeline.commands

        runner = registry.get_runner(runner)()
        runner.run(cfg_json_uri, pipeline, commands, num_splits=splits)


if __name__ == '__main__':
    run()
