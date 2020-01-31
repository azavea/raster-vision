import click

from rastervision.v2.core.main import main

@main.command(
    'predict', short_help='Use a model bundle to predict on new images.')
@click.pass_context
def run_command(ctx):
    profile = ctx.parent.params.get('profile')
    verbose = ctx.parent.params.get('verbose')
    click.echo('hello')

    # call system_init using rv_config
    # build pipeline config
    # raise error if pipeline doesn't have predict as a command, raster source doesn't have uri or uris, or label store doesn't have uri
    # create a single scene based on cli inputs
    # run predict command
