import click

from rastervision.builders import (compute_raster_stats_builder,
                                   make_training_chips_builder, train_builder,
                                   predict_builder, eval_builder)


def _compute_raster_stats(config_uri):
    command = compute_raster_stats_builder.build(config_uri)
    command.run()


@click.command()
@click.argument('config_uri')
def compute_raster_stats(config_uri):
    _compute_raster_stats(config_uri)


def _make_training_chips(config_uri):
    command = make_training_chips_builder.build(config_uri)
    command.run()


@click.command()
@click.argument('config_uri')
def make_training_chips(config_uri):
    _make_training_chips(config_uri)


def _train(config_uri):
    command = train_builder.build(config_uri)
    command.run()


@click.command()
@click.argument('config_uri')
def train(config_uri):
    _train(config_uri)


def _predict(config_uri):
    command = predict_builder.build(config_uri)
    command.run()


@click.command()
@click.argument('config_uri')
def predict(config_uri):
    _predict(config_uri)


def _eval(config_uri):
    command = eval_builder.build(config_uri)
    command.run()


@click.command()
@click.argument('config_uri')
def eval(config_uri):
    _eval(config_uri)


@click.group()
def run():
    pass


run.add_command(compute_raster_stats)
run.add_command(make_training_chips)
run.add_command(train)
run.add_command(predict)
run.add_command(eval)

if __name__ == '__main__':
    run()
